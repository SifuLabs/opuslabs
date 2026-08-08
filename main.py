"""
Video Editing Copilot Agent - Production Ready

This version gracefully handles all dependency issues and provides
a complete working experience with progressive feature loading.
"""

import argparse
import os
import sys
import json
import re
import random
from dataclasses import asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from src.content_strategy import ContentStrategyBuilder
except Exception:
    ContentStrategyBuilder = None

try:
    from src.brand_kits import BRAND_SETTING_KEYS, BrandKitStore
except Exception:
    BRAND_SETTING_KEYS = set()
    BrandKitStore = None

try:
    from src.subtitle_translator import LANGUAGE_ALIASES
    from src.transcript_tools import apply_transcript_corrections, load_transcript_corrections
except Exception:
    LANGUAGE_ALIASES = {}
    apply_transcript_corrections = None
    load_transcript_corrections = None

try:
    from src.job_manager import JobCancelled, JobManager, JobRecord, ProjectRecord
except Exception:
    JobCancelled = RuntimeError
    JobManager = None
    JobRecord = None
    ProjectRecord = None

try:
    from src.oauth_connections import OAuthConnectionService
    from src.publishing import PublishRequest, create_default_publishing_service
except Exception:
    OAuthConnectionService = None
    PublishRequest = None
    create_default_publishing_service = None

try:
    from src.analytics import AnalyticsService
except Exception:
    AnalyticsService = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("⚠️  python-dotenv not installed - environment variables from system only")
    pass

class VideoEditingCopilot:
    """Production-ready Video Editing Copilot with graceful degradation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.available_features = set(['demo_mode', 'conversation'])
        self.moviepy_available = False
        self.ffmpeg_available = False
        self.content_strategy = ContentStrategyBuilder() if ContentStrategyBuilder else None
        self.brand_kits = BrandKitStore() if BrandKitStore else None
        self.job_manager = JobManager() if JobManager else None
        if self.job_manager:
            self.available_features.add('projects_and_jobs')
        self.publishing_service = (
            create_default_publishing_service()
            if create_default_publishing_service
            else None
        )
        self.oauth_connections = (
            OAuthConnectionService(
                store=self.publishing_service.store,
                token_vault=self.publishing_service.token_vault,
            )
            if self.publishing_service and OAuthConnectionService
            else None
        )
        if self.publishing_service:
            self.available_features.add('local_publish_drafts')
        self.analytics_service = (
            AnalyticsService.create_default(
                publication_store=(
                    self.publishing_service.store if self.publishing_service else None
                )
            )
            if AnalyticsService
            else None
        )
        if self.analytics_service:
            self.available_features.add('analytics_feedback')
        
        # Core conversation templates
        self.style_keywords = {
            'funny': ['funny', 'humor', 'comedy', 'laugh', 'hilarious', 'joke'],
            'educational': ['educational', 'learn', 'teach', 'explain', 'tutorial'],
            'energetic': ['energetic', 'exciting', 'hype', 'pump', 'intense'],
            'emotional': ['emotional', 'touching', 'heartfelt', 'moving', 'deep'],
            'viral': ['viral', 'trending', 'hook', 'attention', 'grabbing', 'secret', 'truth', 'controversial', 'surprising'],
            'professional': ['professional', 'business', 'corporate', 'clean']
        }
        
        # Try to load advanced modules
        self._try_load_advanced_modules()
        
        print(f"🎬 Video Editing Copilot Agent Ready!")
        print(f"✅ Available features: {', '.join(sorted(self.available_features))}")
        
        if 'full_processing' not in self.available_features:
            print("💡 Running in demo mode - showing what I can do!")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration safely"""
        config = {
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'output_directory': 'output',
            'temp_directory': 'temp',
            'default_clip_count': 5,
            'default_clip_length': (30, 60),
            'video_format': 'mp4'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                config.update(user_config)
            except Exception:
                pass
        
        return config
    
    def _try_load_advanced_modules(self):
        """Try to load advanced modules with graceful fallback"""
        
        # Check for Gemini API
        if self.config.get('gemini_api_key'):
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.config['gemini_api_key'])
                self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
                self.available_features.add('ai_analysis')
                self.available_features.add('transcription')
                print("✅ Gemini AI integration ready")
            except ImportError:
                print("⚠️  Gemini AI unavailable - install google-genai")
            except Exception as e:
                print(f"⚠️  Gemini setup issue: {e}")
        else:
            print("⚠️  No Gemini API key - set GEMINI_API_KEY environment variable")
        
        # Check for video processing
        video_tools = []
        
        # FFmpeg check
        if self._check_ffmpeg():
            self.ffmpeg_available = True
            video_tools.append('ffmpeg')
            self.available_features.add('video_processing')
        
        # MoviePy check
        try:
            from moviepy import VideoFileClip
            video_tools.append('moviepy')
            self.moviepy_available = True
        except ImportError:
            self.moviepy_available = False
            pass
        
        # Whisper check (with proper error handling)
        try:
            import whisper
            video_tools.append('whisper')
            self.available_features.add('transcription')
        except (ImportError, TypeError) as e:
            print(f"⚠️  Whisper unavailable: {e}")
            pass
        
        if video_tools:
            print(f"✅ Video tools available: {', '.join(video_tools)}")
        
        # Check if we have everything for full processing
        if self.ffmpeg_available and 'transcription' in self.available_features:
            self.available_features.add('full_processing')
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in PATH"""
        import subprocess
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            print("⚠️  FFmpeg not found in PATH.")
            print("   Install it with:  winget install --id Gyan.FFmpeg")
            print("   Then restart your terminal so the PATH is refreshed.")
            return False
        except Exception as e:
            print(f"⚠️  FFmpeg check failed: {e}")
            return False
    
    def process_video_request(self, user_input: str, video_path: Optional[str] = None) -> str:
        """Process video editing request with available capabilities"""
        
        # Parse user preferences
        preferences = self._parse_user_preferences(user_input)
        try:
            preferences = self._apply_brand_kit_preferences(preferences)
        except ValueError as error:
            return f"❌ {error}"
        
        # Check for video file if full processing is available
        if 'full_processing' in self.available_features and not video_path:
            video_path = self._request_video_input()
        
        # Route to appropriate processing method
        if video_path and os.path.exists(video_path) and 'full_processing' in self.available_features:
            return self._process_real_video(video_path, preferences, user_input)
        else:
            return self._process_demo_mode(user_input, preferences)
    
    def _parse_user_preferences(self, user_input: str) -> Dict[str, Any]:
        """Parse user preferences from natural language"""
        preferences = {}
        input_lower = user_input.lower()
        
        # Extract clip count
        clip_patterns = [r'(\d+)\s+clips?', r'make\s+(\d+)', r'create\s+(\d+)', r'generate\s+(\d+)']
        for pattern in clip_patterns:
            match = re.search(pattern, input_lower)
            if match:
                preferences['clip_count'] = int(match.group(1))
                break
        
        # Extract length
        length_patterns = [r'(\d+)\s+seconds?', r'(\d+)s\b', r'(\d+)-(\d+)\s+seconds?']
        for pattern in length_patterns:
            match = re.search(pattern, input_lower)
            if match:
                if len(match.groups()) == 1:
                    length = int(match.group(1))
                    preferences['clip_length_min'] = max(15, length - 5)
                    preferences['clip_length_max'] = min(120, length + 5)
                else:
                    preferences['clip_length_min'] = int(match.group(1))
                    preferences['clip_length_max'] = int(match.group(2))
                break
        
        # Detect style
        for style, keywords in self.style_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['style'] = style
                break

        # Detect target platform for the content launch pack
        platform_keywords = {
            'youtube_shorts': ['youtube shorts', 'shorts', 'youtube'],
            'tiktok': ['tiktok', 'tik tok'],
            'instagram': ['instagram', 'reels', 'ig'],
            'linkedin': ['linkedin'],
        }
        for platform, keywords in platform_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['platform'] = platform
                break

        if any(phrase in input_lower for phrase in ['no captions', 'without captions', 'disable captions']):
            preferences['add_captions'] = False
        elif any(phrase in input_lower for phrase in ['with captions', 'add captions', 'burn captions']):
            preferences['add_captions'] = True

        reframe_keywords = {
            'smart': ['smart crop', 'track face', 'face tracking', 'follow speaker'],
            'split': ['split screen', 'split-screen', 'two speaker layout', 'conversation layout'],
            'blur': ['blur background', 'blurred background', 'blur mode'],
            'crop': ['center crop', 'crop mode', 'fill frame'],
            'fit': ['fit mode', 'black bars', 'show full frame'],
        }
        for reframe_mode, keywords in reframe_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['reframe_mode'] = reframe_mode
                break

        caption_theme_keywords = {
            'bold': ['bold captions', 'viral captions', 'karaoke captions'],
            'clean': ['clean captions', 'professional captions'],
            'minimal': ['minimal captions', 'subtle captions'],
        }
        for caption_theme, keywords in caption_theme_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['caption_theme'] = caption_theme
                break

        position_keywords = {
            'top': ['captions at the top', 'captions on top', 'top captions'],
            'middle': ['captions in the middle', 'center captions', 'middle captions'],
            'bottom': ['captions at the bottom', 'captions on the bottom', 'bottom captions'],
        }
        for caption_position, keywords in position_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['caption_position'] = caption_position
                break

        color_match = re.search(
            r'\bcaption(?:\s+text)?\s+colou?r\s+(#[0-9a-f]{6}|0x[0-9a-f]{6}|[a-z]+)',
            input_lower,
        )
        if color_match:
            color = color_match.group(1)
            preferences['caption_color'] = f"0x{color[1:]}" if color.startswith('#') else color

        size_match = re.search(r'\bcaption(?:\s+font)?\s+size\s+(\d{2,3})\b', input_lower)
        if size_match:
            preferences['caption_font_size'] = min(160, max(24, int(size_match.group(1))))

        quoted_brand = re.search(
            r'\b(?:brand label|watermark)\s+(["\'])(.{1,32}?)\1',
            user_input,
            flags=re.IGNORECASE,
        )
        brand_match = quoted_brand or re.search(
            r'\b(?:brand label|watermark)\s+([a-z0-9][a-z0-9 ._-]{1,31})\s*$',
            user_input,
            flags=re.IGNORECASE,
        )
        if brand_match:
            brand_group = 2 if quoted_brand else 1
            preferences['brand_label'] = brand_match.group(brand_group).strip(' ._-"\'')

        logo_match = re.search(
            r'\b(?:brand logo|logo)\s+(["\'])(.+?)\1',
            user_input,
            flags=re.IGNORECASE,
        )
        if logo_match:
            preferences['brand_logo'] = logo_match.group(2).strip()

        logo_position_keywords = {
            'top-left': ['logo top left', 'logo at the top left'],
            'top-right': ['logo top right', 'logo at the top right'],
            'bottom-left': ['logo bottom left', 'logo at the bottom left'],
            'bottom-right': ['logo bottom right', 'logo at the bottom right'],
        }
        for logo_position, keywords in logo_position_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['brand_logo_position'] = logo_position
                break

        brand_kit_match = re.search(
            r'\b(use|save)\s+brand kit\s+(["\'])(.{1,48}?)\2',
            user_input,
            flags=re.IGNORECASE,
        )
        if brand_kit_match:
            action = brand_kit_match.group(1).lower()
            preferences[f'{action}_brand_kit'] = brand_kit_match.group(3).strip()

        correction_match = re.search(
            r'\breplace\s+(["\'])(.+?)\1\s+with\s+(["\'])(.*?)\3',
            user_input,
            flags=re.IGNORECASE,
        )
        if correction_match:
            preferences['transcript_corrections'] = {
                correction_match.group(2): correction_match.group(4),
            }

        correction_file_match = re.search(
            r'\b(?:transcript corrections|corrections file)\s+(["\'])(.+?\.json)\1',
            user_input,
            flags=re.IGNORECASE,
        )
        if correction_file_match:
            preferences['transcript_corrections_file'] = correction_file_match.group(2).strip()

        if re.search(r'\b(?:subtitles?|translations?)\s+(?:in|to)\b', input_lower):
            subtitle_languages = []
            for language_name, language_code in LANGUAGE_ALIASES.items():
                if re.search(rf'\b{re.escape(language_name)}\b', input_lower):
                    subtitle_languages.append(language_code)
            code_match = re.search(
                r'\b(?:subtitles?|translations?)\s+(?:in|to)\s+([a-z]{2,3}(?:-[a-z0-9]{2,8})?)\b',
                input_lower,
            )
            if code_match and code_match.group(1) not in subtitle_languages:
                subtitle_languages.append(code_match.group(1))
            if subtitle_languages:
                preferences['subtitle_languages'] = subtitle_languages[:5]

        # Detect growth goal
        goal_keywords = {
            'subscribers': ['subscriber', 'subscribers', 'subs', 'grow my channel'],
            'views': ['views', 'reach', 'awareness', 'impressions'],
            'engagement': ['comments', 'engagement', 'likes', 'shares'],
            'sales': ['sales', 'leads', 'clients', 'customers', 'bookings'],
        }
        for goal, keywords in goal_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['goal'] = goal
                break

        # Simple niche extraction. Prefer "about X" over "for subscribers about X".
        niche_match = (
            re.search(r'\babout\s+([a-z0-9][a-z0-9\s\-]{2,40})', input_lower)
            or re.search(r'\b(?:for|targeting)\s+([a-z0-9][a-z0-9\s\-]{2,40})', input_lower)
        )
        if niche_match:
            niche = re.split(r'\b(?:with|from|into|that|and|,)\b', niche_match.group(1))[0].strip()
            if niche:
                preferences['niche'] = niche
        
        # Content type hints
        if any(word in input_lower for word in ['podcast', 'interview', 'talk']):
            preferences['content_type'] = 'conversation'
        elif any(word in input_lower for word in ['presentation', 'lecture']):
            preferences['content_type'] = 'educational'
        
        return preferences

    def _apply_brand_kit_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Load and optionally persist named brand controls."""
        if not BrandKitStore:
            return preferences
        store = getattr(self, 'brand_kits', None) or BrandKitStore()
        resolved = dict(preferences)

        kit_name = resolved.pop('use_brand_kit', None)
        if kit_name:
            saved_settings = store.get(kit_name)
            if saved_settings is None:
                available = ', '.join(store.list_names()) or 'none'
                raise ValueError(
                    f'Brand kit "{kit_name}" was not found. Available kits: {available}.'
                )
            resolved = {**saved_settings, **resolved}
            resolved['brand_kit'] = kit_name

        save_name = resolved.pop('save_brand_kit', None)
        if save_name:
            store.save(
                save_name,
                {key: value for key, value in resolved.items() if key in BRAND_SETTING_KEYS},
            )
            resolved['brand_kit'] = save_name
        return resolved

    def _require_job_manager(self) -> Any:
        if not self.job_manager:
            raise RuntimeError('Persistent project support is unavailable.')
        return self.job_manager

    def _require_publishing_service(self) -> Any:
        if not self.publishing_service:
            raise RuntimeError('Publishing support is unavailable.')
        return self.publishing_service

    def _require_analytics_service(self) -> Any:
        analytics = getattr(self, 'analytics_service', None)
        if not analytics:
            raise RuntimeError('Analytics feedback support is unavailable.')
        return analytics

    def _analytics_candidate_pool_size(
        self,
        requested_count: int,
        platform: str,
    ) -> int:
        analytics = getattr(self, 'analytics_service', None)
        if not analytics:
            return requested_count
        return analytics.candidate_pool_size(requested_count, platform)

    def _rerank_clip_candidates(
        self,
        segments: List[Any],
        preferences: Dict[str, Any],
        limit: int,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        analytics = getattr(self, 'analytics_service', None)
        if not analytics:
            return list(segments)[:limit], {
                'applied': False,
                'observation_count': 0,
                'candidates': [],
            }
        ranked, evidence = analytics.rerank_segments(
            segments,
            platform=preferences.get('platform', 'general'),
            requested_style=preferences.get('style', 'engaging'),
        )
        if evidence.get('applied'):
            print(
                'Analytics reranked clip candidates using '
                f'{evidence.get("observation_count", 0)} observed result(s).'
            )
        return ranked[:limit], evidence

    def approve_clip_for_publishing(
        self,
        clip_path: str,
        approved_by: str,
        note: str = '',
    ) -> Any:
        """Persist approval evidence tied to the current clip fingerprint."""
        return self._require_publishing_service().store.create_approval(
            clip_path,
            approved_by,
            note,
        )

    def submit_publication(self, request: Any) -> Any:
        """Submit an approval-gated provider-neutral publishing request."""
        return self._require_publishing_service().submit(request)

    def create_project(self, name: str, sources: Optional[List[str]] = None) -> Any:
        """Create a durable multi-source project."""
        return self._require_job_manager().create_project(name, sources or [])

    def add_project_source(self, project_id: str, source_path: str) -> Any:
        """Add another source video or URL to a project."""
        return self._require_job_manager().add_source(project_id, source_path)

    def queue_project(self, project_id: str, user_request: str) -> List[Any]:
        """Create one isolated queued job for every source in a project."""
        preferences = self._parse_user_preferences(user_request)
        preferences = self._apply_brand_kit_preferences(preferences)
        for path_key in ('brand_logo', 'transcript_corrections_file'):
            configured_path = preferences.get(path_key)
            if configured_path:
                preferences[path_key] = str(Path(configured_path).expanduser().resolve())
        return self._require_job_manager().enqueue_project(
            project_id,
            user_request,
            preferences,
        )

    def run_next_job(self, project_id: Optional[str] = None) -> Optional[Any]:
        """Claim and process one queued job, persisting its terminal state."""
        manager = self._require_job_manager()
        job = manager.claim_next_job(project_id)
        if job is None:
            return None
        try:
            response = self._process_real_video(
                job.source_path,
                job.preferences,
                job.request_text,
                job_manager=manager,
                job_id=job.id,
            )
            manager.check_cancelled(job.id)
            checkpoint = manager.read_checkpoint(job.id, 'clips') or []
            result = {
                'response': response,
                'clips': checkpoint,
                'output_directory': manager.workspace_directories(job.id)['output'],
            }
            return manager.complete_job(job.id, result)
        except JobCancelled as error:
            return manager.mark_cancelled(job.id, str(error))
        except Exception as error:
            return manager.fail_job(job.id, str(error))

    def run_queued_jobs(
        self,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """Process queued jobs sequentially; multiple processes may call this safely."""
        if limit is not None and limit < 1:
            raise ValueError('Job run limit must be at least one.')
        processed = []
        while limit is None or len(processed) < limit:
            job = self.run_next_job(project_id)
            if job is None:
                break
            processed.append(job)
        return processed
    
    def _request_video_input(self) -> Optional[str]:
        """Request video input from user"""
        print("\n📁 Please provide your video:")
        print("   • Enter file path")
        print("   • Enter YouTube URL")
        print("   • Or drag & drop video file")
        print("   • Or press Enter for demo mode")
        
        try:
            video_input = input("Video source: ").strip()
            
            if not video_input:
                return None
            
            # Check if it's a YouTube URL
            if 'youtube.com' in video_input or 'youtu.be' in video_input:
                return self._download_youtube_video(video_input)
            
            # Clean up the path (remove quotes)
            video_path = video_input.strip('"\'')
            
            if os.path.exists(video_path):
                return video_path
            else:
                print(f"❌ File not found: {video_path}")
                return None
                
        except KeyboardInterrupt:
            print("\n⏹️  Cancelled by user")
            return None
    
    def _download_youtube_video(self, url: str) -> Optional[str]:
        """Download YouTube video using yt-dlp"""
        try:
            import yt_dlp
            
            # Create downloads directory
            downloads_dir = os.path.join(os.getcwd(), 'downloads')
            os.makedirs(downloads_dir, exist_ok=True)
            
            print(f"📥 Downloading from YouTube...")
            print(f"🔗 URL: {url}")
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': os.path.join(downloads_dir, '%(title)s.%(ext)s'),
                'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best quality
                'noplaylist': True,  # Only download single video
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                print(f"🎬 Title: {title}")
                print(f"⏱️ Duration: {duration//60}:{duration%60:02d}")
                
                # Download the video
                print("⬇️ Downloading...")
                ydl.download([url])
                
                # Find the downloaded file
                expected_filename = ydl.prepare_filename(info)
                
                # yt-dlp might change the extension, so check for common formats
                base_name = os.path.splitext(expected_filename)[0]
                possible_extensions = ['.mp4', '.webm', '.mkv', '.avi']
                
                for ext in possible_extensions:
                    potential_file = base_name + ext
                    if os.path.exists(potential_file):
                        print(f"✅ Downloaded: {os.path.basename(potential_file)}")
                        return potential_file
                
                # If exact match not found, look for any video file in downloads
                if os.path.exists(downloads_dir):
                    video_files = [f for f in os.listdir(downloads_dir) 
                                 if f.lower().endswith(('.mp4', '.webm', '.mkv', '.avi'))]
                    if video_files:
                        # Get the most recently created file
                        video_files.sort(key=lambda x: os.path.getctime(os.path.join(downloads_dir, x)), reverse=True)
                        downloaded_file = os.path.join(downloads_dir, video_files[0])
                        print(f"✅ Downloaded: {os.path.basename(downloaded_file)}")
                        return downloaded_file
                
                print("❌ Could not locate downloaded file")
                return None
                
        except ImportError:
            print("❌ yt-dlp not installed!")
            print("💡 Install with: pip install yt-dlp")
            return None
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def _process_real_video_direct(self, video_path: str, preferences: Dict[str, Any], user_input: str) -> str:
        """Process actual video file using the full src/ pipeline"""
        try:
            from src.video_processor import VideoProcessor
            from src.gemini_analyzer import GeminiTranscriptAnalyzer
            from src.clip_generator import ClipGenerator
        except ImportError as e:
            return f"❌ Could not import processing modules: {e}"

        try:
            print(f"🎥 Processing: {os.path.basename(video_path)}")

            # Step 1: Transcribe video → structured dict with real segment timestamps
            print("🎤 Transcribing video...")
            processor = VideoProcessor()
            transcript = processor.transcribe_video(video_path)
            if not transcript or not transcript.get('segments'):
                return "❌ Transcription failed — no segments found. Check FFmpeg and Whisper setup."

            correction_pairs = dict(preferences.get('transcript_corrections', {}))
            correction_path = preferences.get('transcript_corrections_file')
            if correction_path:
                if not load_transcript_corrections:
                    return "❌ Transcript correction helpers are unavailable."
                try:
                    correction_pairs.update(load_transcript_corrections(correction_path))
                except ValueError as error:
                    return f"❌ {error}"
            if correction_pairs:
                if not apply_transcript_corrections:
                    return "❌ Transcript correction helpers are unavailable."
                transcript, correction_count = apply_transcript_corrections(
                    transcript,
                    correction_pairs,
                )
                print(f"✏️ Applied {correction_count} transcript corrections")

            duration = transcript.get('duration', 0) or transcript['segments'][-1]['end']
            seg_count = len(transcript['segments'])
            print(f"📏 Duration: {duration:.1f}s | Segments: {seg_count}")

            # Step 2: AI analysis with accurate timestamps
            clip_count = preferences.get('clip_count', self.config.get('default_clip_count', 5))
            min_len = preferences.get('clip_length_min', 30)
            max_len = preferences.get('clip_length_max', 60)
            style = preferences.get('style', 'engaging')

            print("🧠 Analyzing content for engaging moments...")
            analyzer = GeminiTranscriptAnalyzer()
            candidate_count = self._analytics_candidate_pool_size(
                clip_count,
                preferences.get('platform', 'general'),
            )
            engaging_segments = analyzer.find_engaging_moments(
                transcript,
                target_clips=candidate_count,
                min_length=min_len,
                max_length=max_len,
                style=style
            )
            engaging_segments, _reranking = self._rerank_clip_candidates(
                engaging_segments,
                preferences,
                clip_count,
            )

            if not engaging_segments:
                return "❌ No engaging segments found. Try a different style or check that the video has speech."

            # Step 3: Create clips
            print(f"✂️ Creating {len(engaging_segments)} clips...")
            generator = ClipGenerator()
            settings = {
                'add_captions': preferences.get('add_captions', True),
                'reframe_mode': preferences.get('reframe_mode', 'blur'),
                'caption_theme': preferences.get('caption_theme', 'bold'),
                'caption_position': preferences.get('caption_position', 'bottom'),
                'caption_color': preferences.get('caption_color', 'white'),
                'caption_font_size': preferences.get('caption_font_size'),
                'brand_label': preferences.get('brand_label'),
                'brand_logo': preferences.get('brand_logo'),
                'brand_logo_position': preferences.get('brand_logo_position', 'top-right'),
                'brand_logo_width': preferences.get('brand_logo_width'),
                'brand_logo_opacity': preferences.get('brand_logo_opacity', 0.9),
                'brand_kit': preferences.get('brand_kit'),
                'subtitle_languages': preferences.get('subtitle_languages', []),
                'style': style,
                'platform': preferences.get('platform', 'general'),
                'goal': preferences.get('goal', 'engagement'),
                'niche': preferences.get('niche'),
                'create_thumbnails': True,
                'optimize_for_platform': True,
                'export_subtitles': True,
                'export_manifest': True,
            }
            clips_info = generator.create_clips(video_path, engaging_segments, settings)

            return self._format_success_response(clips_info, preferences)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Processing error: {e}"
    
    def _process_real_video(
        self,
        video_path: str,
        preferences: Dict[str, Any],
        user_input: str,
        job_manager: Optional[Any] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Process directly, or use isolated directories and checkpoints for a job."""
        if not job_manager or not job_id:
            return self._process_real_video_direct(video_path, preferences, user_input)

        from src.clip_generator import ClipGenerator
        from src.gemini_analyzer import EngagingSegment, GeminiTranscriptAnalyzer
        from src.video_processor import VideoProcessor

        manager = job_manager
        directories = manager.workspace_directories(job_id)

        def report(progress: int, stage: str) -> None:
            manager.update_progress(job_id, progress, stage)

        def cancellation_requested() -> bool:
            current_job = manager.get_job(job_id)
            return bool(
                not current_job
                or current_job.cancel_requested
                or current_job.state == 'cancelled'
            )

        report(2, 'Preparing source')
        processor = VideoProcessor()
        processor.temp_dir = directories['temp']
        os.makedirs(processor.temp_dir, exist_ok=True)

        working_video_path = video_path
        if video_path.startswith(('http://', 'https://')):
            source_checkpoint = manager.read_checkpoint(job_id, 'source')
            cached_source = source_checkpoint.get('local_path') if source_checkpoint else None
            if cached_source and os.path.exists(cached_source):
                working_video_path = cached_source
            else:
                processor.temp_dir = directories['input']
                working_video_path = processor.download_video(video_path)
                if not working_video_path:
                    raise RuntimeError('Source download failed.')
                manager.write_checkpoint(
                    job_id,
                    'source',
                    {'original': video_path, 'local_path': working_video_path},
                )
                processor.temp_dir = directories['temp']

        report(8, 'Loading transcript')
        transcript = manager.read_checkpoint(job_id, 'transcript_raw')
        if transcript:
            print('⚡ Reusing transcript checkpoint')
        else:
            print('🎤 Transcribing video...')
            transcript = processor.transcribe_video(working_video_path)
            if not transcript or not transcript.get('segments'):
                raise RuntimeError(
                    'Transcription failed — no segments found. Check FFmpeg and Whisper setup.'
                )
            manager.write_checkpoint(job_id, 'transcript_raw', transcript)
        report(38, 'Transcript ready')

        correction_pairs = dict(preferences.get('transcript_corrections', {}))
        correction_path = preferences.get('transcript_corrections_file')
        if correction_path:
            if not load_transcript_corrections:
                raise RuntimeError('Transcript correction helpers are unavailable.')
            correction_pairs.update(load_transcript_corrections(correction_path))
        if correction_pairs:
            if not apply_transcript_corrections:
                raise RuntimeError('Transcript correction helpers are unavailable.')
            transcript, correction_count = apply_transcript_corrections(
                transcript,
                correction_pairs,
            )
            print(f'✏️ Applied {correction_count} transcript corrections')
        manager.write_checkpoint(job_id, 'transcript', transcript)

        clip_count = preferences.get('clip_count', self.config.get('default_clip_count', 5))
        min_len = preferences.get('clip_length_min', 30)
        max_len = preferences.get('clip_length_max', 60)
        style = preferences.get('style', 'engaging')

        report(42, 'Loading clip analysis')
        segment_checkpoint = manager.read_checkpoint(job_id, 'segments')
        if segment_checkpoint:
            engaging_segments = [EngagingSegment(**segment) for segment in segment_checkpoint]
            print('⚡ Reusing clip-analysis checkpoint')
        else:
            analyzer = GeminiTranscriptAnalyzer()
            candidate_count = self._analytics_candidate_pool_size(
                clip_count,
                preferences.get('platform', 'general'),
            )
            engaging_segments = analyzer.find_engaging_moments(
                transcript,
                target_clips=candidate_count,
                min_length=min_len,
                max_length=max_len,
                style=style,
            )
            engaging_segments, reranking = self._rerank_clip_candidates(
                engaging_segments,
                preferences,
                clip_count,
            )
            if engaging_segments:
                manager.write_checkpoint(
                    job_id,
                    'segments',
                    [asdict(segment) for segment in engaging_segments],
                )
                manager.write_checkpoint(job_id, 'reranking', reranking)
        if not engaging_segments:
            raise RuntimeError(
                'No engaging segments found. Try a different style or verify the source speech.'
            )
        report(62, 'Clip analysis ready')

        generator = ClipGenerator()
        generator.output_dir = directories['output']
        generator.temp_dir = directories['temp']
        os.makedirs(generator.output_dir, exist_ok=True)
        os.makedirs(generator.temp_dir, exist_ok=True)

        def clip_progress(completed: int, total: int) -> None:
            progress = 64 + int((completed / max(1, total)) * 31)
            report(progress, f'Rendered clip {completed} of {total}')

        settings = {
            'add_captions': preferences.get('add_captions', True),
            'reframe_mode': preferences.get('reframe_mode', 'blur'),
            'caption_theme': preferences.get('caption_theme', 'bold'),
            'caption_position': preferences.get('caption_position', 'bottom'),
            'caption_color': preferences.get('caption_color', 'white'),
            'caption_font_size': preferences.get('caption_font_size'),
            'brand_label': preferences.get('brand_label'),
            'brand_logo': preferences.get('brand_logo'),
            'brand_logo_position': preferences.get('brand_logo_position', 'top-right'),
            'brand_logo_width': preferences.get('brand_logo_width'),
            'brand_logo_opacity': preferences.get('brand_logo_opacity', 0.9),
            'brand_kit': preferences.get('brand_kit'),
            'subtitle_languages': preferences.get('subtitle_languages', []),
            'style': style,
            'platform': preferences.get('platform', 'general'),
            'goal': preferences.get('goal', 'engagement'),
            'niche': preferences.get('niche'),
            'create_thumbnails': True,
            'optimize_for_platform': True,
            'export_subtitles': True,
            'export_manifest': True,
            '_cancel_check': cancellation_requested,
            '_progress_callback': clip_progress,
        }
        clips_info = generator.create_clips(
            working_video_path,
            engaging_segments,
            settings,
        )
        report(96, 'Finalizing output package')
        if not clips_info:
            raise RuntimeError('No clips were rendered successfully.')
        manager.write_checkpoint(job_id, 'clips', clips_info)
        return self._format_success_response(clips_info, preferences)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using MoviePy"""
        try:
            if self.moviepy_available:
                from moviepy import VideoFileClip
                with VideoFileClip(video_path) as video:
                    return video.duration
        except Exception as e:
            print(f"⚠️  Could not get video duration: {e}")
        
        # Fallback estimate
        return 600.0
    
    def _transcribe_video(self, video_path: str) -> str:
        """Transcribe video using available tools"""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(video_path)
            return result["text"]
        except:
            # Fallback to simple transcript
            return f"[Transcript of {os.path.basename(video_path)} - content analysis based on filename and duration]"
    
    def _analyze_with_gemini(self, transcript: str, preferences: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        """Analyze content with Gemini AI"""
        try:
            style = preferences.get('style', 'engaging')
            clip_count = preferences.get('clip_count', 5)
            
            # Enhanced prompt for better content analysis
            prompt = f"""
            As a professional video editor and social media expert, analyze this transcript to create {clip_count} highly engaging {style} short-form clips.

            User Request: {user_request}
            Target Style: {style}
            Desired Clips: {clip_count}

            Transcript:
            {transcript[:6000]}

            Please identify the most compelling moments that would work best as short-form content by analyzing:
            
            1. **Hook Potential**: Look for surprising statements, questions, or dramatic moments
            2. **Emotional Peaks**: Find moments of high energy, laughter, insight, or strong emotion  
            3. **Complete Thoughts**: Ensure each clip has a beginning, middle, and satisfying conclusion
            4. **Visual Interest**: Consider moments that likely have dynamic visuals or gestures
            5. **Shareability**: Pick moments people would want to share or comment on
            
            For each recommended clip, provide:
            - A compelling hook/title (8-12 words max)
            - The exact key phrase or sentence that makes it engaging
            - Why this moment would perform well on social media
            - Estimated optimal length (15-60 seconds)
            - The type of audience reaction expected
            
            Focus on finding moments that:
            - Start with immediate impact (first 3 seconds crucial)
            - Have clear value or entertainment
            - End with satisfaction or curiosity
            - Work without prior context
            
            Return your analysis as structured recommendations focusing on the most viral-worthy segments.
            """
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )
            ai_text = response.text
            
            # Parse AI insights and extract practical suggestions
            suggestions = self._parse_ai_suggestions(ai_text, transcript, clip_count)
            
            analysis = {
                'style': style,
                'suggested_clips': suggestions,
                'ai_insights': ai_text,
                'content_analysis': {
                    'high_engagement_moments': len(suggestions),
                    'recommended_style': style,
                    'content_type': self._detect_content_type(transcript)
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return self._create_simple_analysis("", preferences)
    
    def _parse_ai_suggestions(self, ai_text: str, transcript: str, clip_count: int) -> List[Dict[str, Any]]:
        """Parse AI suggestions into actionable clip data"""
        suggestions = []
        
        try:
            # Look for key phrases that indicate engaging moments
            transcript_sentences = transcript.split('. ')
            ai_text_lower = ai_text.lower()
            
            # AI-identified patterns
            hook_indicators = [
                'surprising', 'shocking', 'amazing', 'unbelievable', 'incredible',
                'secret', 'revealed', 'truth', 'mistake', 'wrong', 'right way',
                'never knew', 'didn\'t know', 'most people', 'everyone should',
                'game changer', 'life changing', 'mind blowing'
            ]
            
            # Find sentences with high engagement potential
            engaging_sentences = []
            for i, sentence in enumerate(transcript_sentences):
                sentence_lower = sentence.lower()
                score = 0
                
                # Score based on AI indicators
                for indicator in hook_indicators:
                    if indicator in sentence_lower or indicator in ai_text_lower:
                        score += 2
                
                # Score based on sentence characteristics
                if '?' in sentence:  # Questions are engaging
                    score += 1
                if len(sentence.split()) > 8:  # Substantial content
                    score += 1
                if any(word in sentence_lower for word in ['you', 'your', 'we']):
                    score += 1  # Direct address
                
                if score > 0:
                    engaging_sentences.append({
                        'sentence': sentence.strip(),
                        'index': i,
                        'score': score,
                        'estimated_time': i * 3  # Rough estimate: 3 seconds per sentence
                    })
            
            # Sort by engagement score
            engaging_sentences.sort(key=lambda x: x['score'], reverse=True)
            
            # Create clips from top moments
            selected_count = min(len(engaging_sentences), clip_count)
            for i in range(selected_count):
                moment = engaging_sentences[i]
                
                # Create compelling title from sentence
                title = self._create_clip_title(moment['sentence'], i+1)
                
                suggestion = {
                    'clip_number': i + 1,
                    'title': title,
                    'key_phrase': moment['sentence'][:100] + '...' if len(moment['sentence']) > 100 else moment['sentence'],
                    'engagement_reason': self._explain_engagement(moment['sentence']),
                    'estimated_start': max(0, moment['estimated_time'] - 5),  # Start 5s before
                    'recommended_duration': min(60, max(20, len(moment['sentence'].split()) * 2)),
                    'engagement_score': moment['score']
                }
                suggestions.append(suggestion)
            
            # Fill remaining slots if needed
            while len(suggestions) < clip_count:
                i = len(suggestions)
                suggestions.append({
                    'clip_number': i + 1,
                    'title': f'Engaging Moment {i + 1}',
                    'key_phrase': 'Additional compelling content identified',
                    'engagement_reason': 'Strong content value for social media',
                    'estimated_start': i * 60,  # Spread across video
                    'recommended_duration': 30,
                    'engagement_score': 3
                })
            
        except Exception as e:
            print(f"⚠️  Error parsing AI suggestions: {e}")
            # Fallback to basic suggestions
            for i in range(clip_count):
                suggestions.append({
                    'clip_number': i + 1,
                    'title': f'Engaging Clip {i + 1}',
                    'key_phrase': 'AI-identified compelling moment',
                    'engagement_reason': 'High potential for social media engagement',
                    'estimated_start': i * 60,
                    'recommended_duration': 30,
                    'engagement_score': 5
                })
        
        return suggestions
    
    def _create_clip_title(self, sentence: str, clip_num: int) -> str:
        """Create engaging title from sentence content"""
        sentence = sentence.strip()
        
        # Extract key concepts
        words = sentence.lower().split()
        
        # Look for power words
        power_words = ['secret', 'truth', 'mistake', 'wrong', 'right', 'best', 'worst', 'never', 'always', 'everyone', 'nobody']
        title_words = []
        
        for word in words:
            if word in power_words:
                title_words.append(word.title())
        
        # If we found power words, build title around them
        if title_words:
            if 'secret' in sentence.lower():
                return f"The Secret That {clip_num}"
            elif 'mistake' in sentence.lower():
                return f"Biggest Mistake #{clip_num}"
            elif 'truth' in sentence.lower():
                return f"Truth About This"
            elif any(w in sentence.lower() for w in ['never', 'nobody']):
                return f"Nobody Talks About This"
        
        # Fallback to sentence start
        first_words = ' '.join(sentence.split()[:4])
        return f"{first_words}..." if len(sentence.split()) > 4 else first_words
    
    def _explain_engagement(self, sentence: str) -> str:
        """Explain why a moment is engaging"""
        sentence_lower = sentence.lower()
        
        if '?' in sentence:
            return "Questions drive curiosity and comments"
        elif any(word in sentence_lower for word in ['secret', 'truth', 'reveal']):
            return "Exclusive insights create shareability"
        elif any(word in sentence_lower for w in ['mistake', 'wrong', 'error']):
            return "Learning from mistakes resonates with audiences"
        elif any(word in sentence_lower for word in ['you', 'your']):
            return "Direct address increases personal connection"
        else:
            return "Compelling content with strong social media potential"
    
    def _detect_content_type(self, transcript: str) -> str:
        """Detect the type of content from transcript"""
        transcript_lower = transcript.lower()
        
        if any(word in transcript_lower for word in ['interview', 'conversation', 'talk', 'discussion']):
            return 'interview'
        elif any(word in transcript_lower for word in ['learn', 'teach', 'explain', 'tutorial', 'lesson']):
            return 'educational'
        elif any(word in transcript_lower for word in ['story', 'happened', 'experience', 'remember']):
            return 'storytelling'
        elif any(word in transcript_lower for word in ['business', 'strategy', 'marketing', 'sales']):
            return 'business'
        else:
            return 'general'
    
    def _create_simple_analysis(self, video_path: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Create simple analysis without AI"""
        style = preferences.get('style', 'engaging')
        clip_count = preferences.get('clip_count', 5)
        
        return {
            'style': style,
            'clip_count': clip_count,
            'content_type': preferences.get('content_type', 'general'),
            'suggested_clips': []
        }
        """Create simple analysis without AI"""
        style = preferences.get('style', 'engaging')
        clip_count = preferences.get('clip_count', 5)
        
        return {
            'style': style,
            'clip_count': clip_count,
            'content_type': preferences.get('content_type', 'general'),
            'suggested_clips': []
        }
    
    def _plan_clips(self, video_path: str, analysis: Dict[str, Any], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan clips based on analysis with improved accuracy"""
        clip_count = preferences.get('clip_count', analysis.get('clip_count', 5))
        style = preferences.get('style', analysis.get('style', 'engaging'))
        
        clips = []
        duration = self._get_video_duration(video_path)
        
        # Use AI suggestions if available
        ai_suggestions = analysis.get('suggested_clips', [])
        
        if ai_suggestions:
            print(f"🧠 Using AI-identified engaging moments...")
            
            for i, suggestion in enumerate(ai_suggestions[:clip_count]):
                # Use AI timing suggestions but validate them
                suggested_start = suggestion.get('estimated_start', i * (duration / clip_count))
                suggested_duration = suggestion.get('recommended_duration', 30)
                
                # Ensure valid timing
                start_time = max(0, min(suggested_start, duration - suggested_duration - 1))
                clip_duration = min(suggested_duration, duration - start_time - 1)
                clip_duration = max(15, clip_duration)  # Minimum 15 seconds
                
                clip = {
                    'clip_number': i + 1,
                    'start_time': start_time,
                    'duration': clip_duration,
                    'title': suggestion.get('title', f"{style.title()} Moment {i + 1}"),
                    'style': style,
                    'output_path': f"output/clip_{i+1:02d}_{style}_{suggestion.get('engagement_score', 5)}.mp4",
                    'engagement_score': suggestion.get('engagement_score', 5),
                    'ai_reasoning': suggestion.get('engagement_reason', 'AI-identified engaging content'),
                    'key_phrase': suggestion.get('key_phrase', 'Compelling moment')
                }
                clips.append(clip)
                
                print(f"   📱 Clip {i+1}: {clip['title']}")
                print(f"      🎯 Key: {clip['key_phrase'][:60]}...")
                print(f"      ⏰ {start_time:.1f}s - {start_time + clip_duration:.1f}s")
        
        else:
            print(f"📊 Creating evenly distributed clips...")
            # Fallback to evenly spaced clips with better distribution
            segment_length = duration / (clip_count + 1)
            
            for i in range(clip_count):
                # Better distribution - avoid very beginning and end
                start_time = segment_length * (i + 1) - 30
                start_time = max(30, min(start_time, duration - 60))  # Keep away from edges
                
                clip_length = random.uniform(
                    preferences.get('clip_length_min', 30),
                    preferences.get('clip_length_max', 60)
                )
                
                # Ensure clip fits in video
                clip_length = min(clip_length, duration - start_time - 1)
                
                clip = {
                    'clip_number': i + 1,
                    'start_time': start_time,
                    'duration': clip_length,
                    'title': f"{style.title()} Moment {i + 1}",
                    'style': style,
                    'output_path': f"output/clip_{i+1:02d}_{style}.mp4",
                    'engagement_score': round(random.uniform(6.0, 8.5), 1)
                }
                clips.append(clip)
        
        return clips
    
    def _create_clips(self, video_path: str, clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Actually create video clips using MoviePy"""
        created_clips = []
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        if not self.moviepy_available:
            print("❌ MoviePy not available - cannot create actual clips")
            return clips
        
        video = None
        try:
            from moviepy import VideoFileClip
            
            print(f"🎥 Loading video: {os.path.basename(video_path)}")
            video = VideoFileClip(video_path)
            
            print(f"✅ Video loaded successfully!")
            print(f"📏 Duration: {video.duration:.1f} seconds")
            print(f"📐 Size: {video.size}")
            
            for clip in clips:
                clip_segment = None
                clip_vertical = None
                try:
                    start_time = clip['start_time']
                    duration = clip['duration']
                    output_path = clip['output_path']
                    
                    print(f"✂️ Creating clip {clip['clip_number']}/{len(clips)}...")
                    
                    # Ensure start time is valid
                    start_time = max(0, min(start_time, video.duration - 1))
                    end_time = min(start_time + duration, video.duration - 0.1)
                    
                    if end_time <= start_time:
                        raise Exception(f"Invalid time range: {start_time}-{end_time}")
                    
                    # Extract the clip segment
                    clip_segment = video.subclipped(start_time, end_time)
                    
                    if clip_segment is None:
                        raise Exception("Failed to create clip segment")
                    
                    # Convert to vertical format (9:16)
                    clip_vertical = self._make_vertical_clip(clip_segment)
                    
                    if clip_vertical is None:
                        raise Exception("Failed to convert to vertical format")
                    
                    # Save the clip with better error handling
                    print(f"💾 Saving: {os.path.basename(output_path)}")
                    
                    # Use safer write parameters
                    clip_vertical.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        verbose=False,
                        logger=None
                    )
                    
                    # Verify file was created
                    if not os.path.exists(output_path):
                        raise Exception("Output file was not created")
                    
                    # Update clip info
                    clip['created'] = True
                    clip['file_size'] = os.path.getsize(output_path)
                    clip['actual_duration'] = clip_vertical.duration
                    
                    created_clips.append(clip)
                    print(f"✅ Clip {clip['clip_number']} created successfully")
                    
                except Exception as e:
                    print(f"❌ Error creating clip {clip['clip_number']}: {e}")
                    clip['created'] = False
                    clip['error'] = str(e)
                    created_clips.append(clip)
                
                finally:
                    # Always clean up clip objects to prevent memory leaks
                    if clip_segment is not None:
                        try:
                            clip_segment.close()
                        except:
                            pass
                    if clip_vertical is not None:
                        try:
                            clip_vertical.close()
                        except:
                            pass
                    
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            # Return clips with error info
            for clip in clips:
                if clip not in created_clips:
                    clip['created'] = False
                    clip['error'] = str(e)
                    created_clips.append(clip)
        
        finally:
            # Always clean up main video
            if video is not None:
                try:
                    video.close()
                except:
                    pass
        
        return created_clips
    
    def _make_vertical_clip(self, clip):
        """Convert clip to vertical 9:16 format"""
        target_width = 1080
        target_height = 1920
        
        # Resize to fit height
        clip_resized = clip.resized(height=target_height)
        
        # If too wide, crop from center
        if clip_resized.w > target_width:
            x_center = clip_resized.w / 2
            x1 = int(x_center - target_width / 2)
            x2 = int(x_center + target_width / 2)
            clip_final = clip_resized.cropped(x1=x1, x2=x2)
        else:
            clip_final = clip_resized
        
        return clip_final
    
    def _process_demo_mode(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """Process in demo mode"""
        # Use the working standalone demo logic
        return self._generate_demo_response(user_input, preferences)
    
    def _generate_demo_response(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """Generate comprehensive demo response"""
        
        # Merge with defaults
        settings = {
            'clip_count': 5,
            'clip_length_min': 30,
            'clip_length_max': 60,
            'style': 'engaging'
        }
        settings.update(preferences)
        
        # Generate enthusiastic response
        processing_phrases = [
            "Perfect! Let me work my magic ✨",
            "Got it! Time to create something amazing 🎬",
            "Excellent choice! Processing your video now 🚀"
        ]
        
        phrase = random.choice(processing_phrases)
        
        response = f"{phrase}\n\n"
        response += "📊 **Here's the plan:**\n"
        response += f"• Creating {settings['clip_count']} clips\n"
        response += f"• Length: {settings['clip_length_min']}-{settings['clip_length_max']} seconds each\n"
        response += f"• Style: {settings['style'].title()}\n"
        response += "• Format: Vertical (9:16) with captions\n\n"
        
        # Simulate processing
        response += "🎤 Analyzing your video content...\n"
        response += "🧠 AI finding engaging moments and hooks...\n"
        response += "✂️ Creating optimized clips...\n\n"
        
        # Create mock clips
        clips = self._create_demo_clips(settings)
        
        response += "🎉 **Your clips are ready! Here's what I created:**\n\n"
        
        for clip in clips:
            response += f"📱 **Clip {clip['clip_number']}**: {clip['title']}\n"
            response += f"   • Duration: {clip['duration']} seconds\n"
            response += f"   • Hook: \"{clip['hook']}\"\n"
            response += f"   • Engagement Score: {clip['engagement_score']}/10\n"
            response += f"   • File: {clip['output_path']}\n"
            response += self._format_launch_pack(clip)
            response += "\n"
        
        # Add pro tips and setup info
        response += "💡 **Pro Tips:**\n"
        response += "• Post during peak hours (7-9 PM)\n"
        response += "• Use trending hashtags for your niche\n"
        response += "• Add compelling captions with calls-to-action\n"
        response += "• Cross-post to multiple platforms!\n\n"
        
        if 'full_processing' not in self.available_features:
            response += "🔧 **To enable full video processing:**\n"
            response += "1. Set GEMINI_API_KEY environment variable\n"
            response += "2. Install FFmpeg: https://ffmpeg.org/download.html\n"
            response += "3. Install dependencies: pip install google-genai openai-whisper moviepy\n"
            response += "4. Run: python main_production.py\n"
        
        return response
    
    def _create_demo_clips(self, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create realistic demo clips"""
        style = settings['style']
        clip_count = settings['clip_count']
        
        # Style-specific content
        templates = {
            'funny': {
                'titles': ['This Had Me Dying', 'Comedy Gold', 'Hilarious Moment'],
                'hooks': ['You won\'t believe what happened next...', 'This is absolutely hilarious']
            },
            'educational': {
                'titles': ['Mind-Blowing Insight', 'Learn This Today', 'Game-Changing Tip'],
                'hooks': ['Here\'s what most people don\'t know...', 'This will change how you think...']
            },
            'viral': {
                'titles': ['This is Going Viral', 'Everyone Needs to See This', 'Trending Right Now'],
                'hooks': ['This is about to blow up...', 'Everyone\'s talking about this...']
            },
            'professional': {
                'titles': ['Key Business Insight', 'Professional Tip', 'Industry Secret'],
                'hooks': ['Here\'s the strategy that works...', 'This is what successful people do...']
            }
        }
        
        style_templates = templates.get(style, templates['viral'])
        
        clips = []
        for i in range(clip_count):
            avg_length = (settings['clip_length_min'] + settings['clip_length_max']) / 2
            duration = round(avg_length + random.uniform(-5, 5), 1)
            
            clip = {
                'clip_number': i + 1,
                'title': random.choice(style_templates['titles']) + f" {i + 1}",
                'hook': random.choice(style_templates['hooks']),
                'duration': max(15, duration),
                'output_path': f"output/clip_{i+1:02d}_{style}.mp4",
                'engagement_score': round(random.uniform(7.0, 9.5), 1)
            }
            if self.content_strategy:
                clip['content_package'] = self.content_strategy.build_for_clip(clip, settings)
            clips.append(clip)
        
        return clips

    def _format_launch_pack(self, clip: Dict[str, Any]) -> str:
        """Format compact publish guidance for a clip."""
        package = clip.get('content_package')
        if not package:
            return ""

        lines = [
            f"   • Publish title: {package['primary_title']}",
            f"   • Caption: {package['short_caption']}",
            f"   • Hashtags: {' '.join(package['hashtags'][:6])}",
            f"   • Thumbnail text: {package['thumbnail_text']}",
            f"   • Best window: {package['posting_window']}",
            f"   • Comment prompt: {package['engagement_question']}",
            f"   • CTA: {package['call_to_action']}",
        ]
        return "\n".join(lines) + "\n"
    
    def _format_success_response(self, clips: List[Dict[str, Any]], preferences: Dict[str, Any]) -> str:
        """Format response for successful video processing"""
        response = "🎉 **Video Processing Complete!**\n\n"
        
        successful_clips = [c for c in clips if c.get('output_path') and os.path.exists(c['output_path'])]
        failed_clips = [c for c in clips if c not in successful_clips]
        
        if successful_clips:
            response += f"✅ **{len(successful_clips)} clips created successfully:**\n\n"
            
            for clip in successful_clips:
                size_kb = os.path.getsize(clip['output_path']) // 1024 if os.path.exists(clip.get('output_path', '')) else 0
                response += f"📱 **Clip {clip.get('clip_number', '?')}**: {clip.get('title', 'Untitled')}\n"
                response += f"   • Duration: {clip.get('duration', 0):.1f}s  ({clip.get('start_time', 0):.1f}s → {clip.get('end_time', 0):.1f}s)\n"
                response += f"   • Hook: \"{clip.get('hook', '')}\"\n"
                response += f"   • Engagement Score: {clip.get('engagement_score', 0):.1f}/10\n"
                response += f"   • File: {clip['output_path']}  ({size_kb}KB)\n"
                if clip.get('hashtags'):
                    response += f"   • Tags: {' '.join(clip['hashtags'][:5])}\n"
                response += self._format_launch_pack(clip)
                response += "\n"
        
        if failed_clips:
            response += f"⚠️  **{len(failed_clips)} clips had issues:**\n"
            for clip in failed_clips:
                response += f"   • Clip {clip.get('clip_number', '?')}: {clip.get('error', 'Unknown error')}\n"
        
        response += "\n🚀 **Your clips are ready for social media!**\n"
        return response
    
    def _format_plan_response(self, clips: List[Dict[str, Any]], preferences: Dict[str, Any]) -> str:
        """Format response showing clip plan"""
        response = "📋 **Clip Plan Created!**\n\n"
        
        for clip in clips:
            response += f"📱 **Clip {clip['clip_number']}**: {clip['title']}\n"
            response += f"   • Start: {clip['start_time']:.1f}s\n"
            response += f"   • Duration: {clip['duration']:.1f}s\n"
            response += f"   • Style: {clip['style']}\n\n"
        
        response += "💡 **To create actual clips:**\n"
        response += "• Install FFmpeg for video processing\n"
        response += "• Run with full processing enabled\n"
        
        return response
    
    def run_interactive_mode(self):
        """Run interactive conversational mode"""
        print("\n🎬 **Video Editing Copilot Agent**")
        print("=" * 50)
        print("Transform your long-form videos into viral short clips!")
        
        # Show capabilities
        print(f"\n✅ **Available Features:**")
        for feature in sorted(self.available_features):
            print(f"   • {feature.replace('_', ' ').title()}")
        
        print("\nI'll help you create engaging short-form content for:")
        print("   🎵 TikTok • 📱 Instagram Reels • 🩳 YouTube Shorts")
        
        print("\nType 'help' for examples, 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("💬 What would you like me to do? ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using Video Editing Copilot! Create amazing content!")
                    break
                
                if user_input.lower() in ['help', '?']:
                    self._show_interactive_help()
                    continue
                
                # Process the request
                response = self.process_video_request(user_input)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using Video Editing Copilot!")
                break
    
    def _show_interactive_help(self):
        """Show help for interactive mode"""
        print("\n💡 **Example requests:**")
        print('• "Turn my podcast into 5 funny clips"')
        print('• "Make 3 educational clips, 45 seconds each"')
        print('• "Create viral content from this presentation"')
        print('• "Generate professional clips for LinkedIn"')
        
        print("\n🎯 **I understand:**")
        print("• Number of clips (1-10)")
        print("• Clip length (15-120 seconds)")
        print("• Style: funny, educational, viral, professional, energetic")
        print("• Content type: podcast, interview, presentation, lecture")
        
        if 'full_processing' in self.available_features:
            print("\n🎬 **Full processing available!**")
            print("I can analyze your actual video files and create clips.")
        else:
            print("\n📋 **Demo mode active**")
            print("I'll show you exactly what I would create.")
            print("Set up full processing for actual video editing!")
        
        print()

def _run_management_cli(copilot: VideoEditingCopilot, arguments: List[str]) -> int:
    """Run persistent project and job management commands."""
    parser = argparse.ArgumentParser(prog='python main.py')
    command_parsers = parser.add_subparsers(dest='resource', required=True)

    project_parser = command_parsers.add_parser('project', help='Manage durable projects')
    project_commands = project_parser.add_subparsers(dest='action', required=True)
    create_project = project_commands.add_parser('create', help='Create a project')
    create_project.add_argument('name')
    create_project.add_argument('sources', nargs='*')
    project_commands.add_parser('list', help='List projects')
    show_project = project_commands.add_parser('show', help='Show a project and its sources')
    show_project.add_argument('project_id')
    add_source = project_commands.add_parser('add-source', help='Add project sources')
    add_source.add_argument('project_id')
    add_source.add_argument('sources', nargs='+')

    job_parser = command_parsers.add_parser('job', help='Manage queued processing jobs')
    job_commands = job_parser.add_subparsers(dest='action', required=True)
    enqueue_job = job_commands.add_parser('enqueue', help='Queue all sources in a project')
    enqueue_job.add_argument('project_id')
    enqueue_job.add_argument('request', nargs='+')
    list_jobs = job_commands.add_parser('list', help='List jobs')
    list_jobs.add_argument('--project', dest='project_id')
    list_jobs.add_argument('--state', choices=sorted(['queued', 'running', 'completed', 'failed', 'cancelled']))
    run_jobs = job_commands.add_parser('run', help='Process queued jobs')
    run_jobs.add_argument('--project', dest='project_id')
    run_jobs.add_argument('--limit', type=int)
    cancel_job = job_commands.add_parser('cancel', help='Request job cancellation')
    cancel_job.add_argument('job_id')
    retry_job = job_commands.add_parser('retry', help='Retry a failed or cancelled job')
    retry_job.add_argument('job_id')
    recover_jobs = job_commands.add_parser('recover', help='Recover stale running jobs')
    recover_jobs.add_argument('--stale-seconds', type=int)
    recover_jobs.add_argument('--force', action='store_true')
    job_events = job_commands.add_parser('events', help='Show a job event timeline')
    job_events.add_argument('job_id')

    publish_parser = command_parsers.add_parser('publish', help='Approve and submit clips')
    publish_commands = publish_parser.add_subparsers(dest='action', required=True)
    publish_commands.add_parser('providers', help='List publishing capabilities')
    approve_publish = publish_commands.add_parser('approve', help='Approve exact clip bytes')
    approve_publish.add_argument('clip_path')
    approve_publish.add_argument('--by', required=True, dest='approved_by')
    approve_publish.add_argument('--note', default='')
    submit_publish = publish_commands.add_parser('submit', help='Submit a provider request')
    submit_publish.add_argument('clip_path')
    submit_publish.add_argument('--approval', required=True, dest='approval_id')
    submit_publish.add_argument('--provider', default='local')
    submit_publish.add_argument('--platform', default='general')
    submit_publish.add_argument('--mode', choices=['draft', 'publish', 'schedule'], default='draft')
    submit_publish.add_argument('--title', required=True)
    submit_publish.add_argument('--caption', default='')
    submit_publish.add_argument('--hashtag', action='append', default=[], dest='hashtags')
    submit_publish.add_argument('--privacy', choices=['private', 'unlisted', 'public'], default='private')
    submit_publish.add_argument('--account', dest='account_id')
    submit_publish.add_argument('--scheduled-at')
    submit_publish.add_argument('--idempotency-key')
    list_publications = publish_commands.add_parser('list', help='List publication records')
    list_publications.add_argument(
        '--state',
        choices=['submitting', 'drafted', 'published', 'scheduled', 'failed'],
    )
    show_publication = publish_commands.add_parser('show', help='Show a publication record')
    show_publication.add_argument('publication_id')

    account_parser = command_parsers.add_parser('account', help='Inspect publishing accounts')
    account_commands = account_parser.add_subparsers(dest='action', required=True)
    list_accounts = account_commands.add_parser('list', help='List connected account metadata')
    list_accounts.add_argument('--provider')
    account_commands.add_parser('oauth-providers', help='List installed OAuth adapters')
    disconnect_account = account_commands.add_parser('disconnect', help='Delete an account and its tokens')
    disconnect_account.add_argument('account_id')

    analytics_parser = command_parsers.add_parser(
        'analytics',
        help='Import observed performance and inspect feedback',
    )
    analytics_commands = analytics_parser.add_subparsers(dest='action', required=True)
    import_analytics = analytics_commands.add_parser(
        'import',
        help='Import a UTF-8 JSON or CSV metrics export',
    )
    import_analytics.add_argument('source_path')
    report_analytics = analytics_commands.add_parser(
        'report',
        help='Compare predicted and observed performance',
    )
    report_analytics.add_argument('--platform')
    report_analytics.add_argument('--style')
    list_analytics = analytics_commands.add_parser(
        'list',
        help='List normalized performance observations',
    )
    list_analytics.add_argument('--platform')
    list_analytics.add_argument('--style')
    list_analytics.add_argument('--limit', type=int, default=100)
    analytics_commands.add_parser('schema', help='Show the provider-neutral import schema')

    options = parser.parse_args(arguments)
    manager = (
        copilot._require_job_manager()
        if options.resource in {'project', 'job'}
        else None
    )

    if options.resource == 'project':
        if options.action == 'create':
            project = copilot.create_project(options.name, options.sources)
            payload = {
                'project': project.to_dict(),
                'sources': [source.to_dict() for source in manager.list_sources(project.id)],
            }
        elif options.action == 'list':
            payload = [
                {
                    **project.to_dict(),
                    'source_count': len(manager.list_sources(project.id)),
                    'job_count': len(manager.list_jobs(project_id=project.id)),
                }
                for project in manager.list_projects()
            ]
        elif options.action == 'show':
            project = manager.get_project(options.project_id)
            if not project:
                raise ValueError(f'Project not found: {options.project_id}')
            payload = {
                'project': project.to_dict(),
                'sources': [source.to_dict() for source in manager.list_sources(project.id)],
                'jobs': [job.to_dict() for job in manager.list_jobs(project_id=project.id)],
            }
        else:
            payload = [
                copilot.add_project_source(options.project_id, source).to_dict()
                for source in options.sources
            ]
    elif options.resource == 'publish':
        publishing = copilot._require_publishing_service()
        if options.action == 'providers':
            payload = [
                capabilities.to_dict()
                for capabilities in publishing.registry.list_capabilities()
            ]
        elif options.action == 'approve':
            payload = copilot.approve_clip_for_publishing(
                options.clip_path,
                options.approved_by,
                options.note,
            ).to_dict()
        elif options.action == 'submit':
            payload = copilot.submit_publication(PublishRequest(
                provider=options.provider,
                platform=options.platform,
                mode=options.mode,
                clip_path=options.clip_path,
                approval_id=options.approval_id,
                title=options.title,
                caption=options.caption,
                hashtags=options.hashtags,
                privacy=options.privacy,
                account_id=options.account_id,
                scheduled_at=options.scheduled_at,
                idempotency_key=options.idempotency_key,
            )).to_dict()
        elif options.action == 'list':
            payload = [
                publication.to_dict()
                for publication in publishing.store.list_publications(options.state)
            ]
        else:
            publication = publishing.store.get_publication(options.publication_id)
            if not publication:
                raise ValueError(f'Publication not found: {options.publication_id}')
            payload = publication.to_dict()
    elif options.resource == 'account':
        publishing = copilot._require_publishing_service()
        if options.action == 'list':
            payload = [
                account.to_dict()
                for account in publishing.store.list_accounts(options.provider)
            ]
        elif options.action == 'oauth-providers':
            payload = (
                copilot.oauth_connections.adapters.list_providers()
                if copilot.oauth_connections
                else []
            )
        else:
            if not copilot.oauth_connections:
                raise RuntimeError('OAuth connection support is unavailable.')
            payload = {'disconnected': copilot.oauth_connections.disconnect(options.account_id)}
    elif options.resource == 'analytics':
        analytics = copilot._require_analytics_service()
        if options.action == 'import':
            payload = analytics.import_file(options.source_path).to_dict()
        elif options.action == 'report':
            payload = analytics.build_report(options.platform, options.style)
        elif options.action == 'list':
            payload = [
                observation.to_dict()
                for observation in analytics.store.list_observations(
                    platform=options.platform,
                    style=options.style,
                    limit=options.limit,
                )
            ]
        else:
            payload = analytics.import_schema()
    elif options.action == 'enqueue':
        jobs = copilot.queue_project(options.project_id, ' '.join(options.request))
        payload = [job.to_dict() for job in jobs]
    elif options.action == 'list':
        payload = [
            job.to_dict()
            for job in manager.list_jobs(options.project_id, options.state)
        ]
    elif options.action == 'run':
        payload = [
            job.to_dict()
            for job in copilot.run_queued_jobs(options.project_id, options.limit)
        ]
    elif options.action == 'cancel':
        payload = manager.request_cancel(options.job_id).to_dict()
    elif options.action == 'retry':
        payload = manager.retry_job(options.job_id).to_dict()
    elif options.action == 'recover':
        payload = manager.recover_interrupted_jobs(
            stale_after_seconds=options.stale_seconds,
            force=options.force,
        )
    else:
        payload = manager.list_events(options.job_id)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """Main entry point."""
    copilot = VideoEditingCopilot()
    try:
        if len(sys.argv) > 1 and sys.argv[1] in {
            'project', 'job', 'publish', 'account', 'analytics'
        }:
            return _run_management_cli(copilot, sys.argv[1:])
        if len(sys.argv) > 1:
            user_request = " ".join(sys.argv[1:])
            response = copilot.process_video_request(user_request)
            print(response)
        else:
            copilot.run_interactive_mode()
        return 0
    except (RuntimeError, ValueError) as error:
        print(f'❌ {error}')
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
