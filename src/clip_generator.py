"""
Clip Generation Module

Handles video cutting, reformatting, and caption overlay for social media clips.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import numpy as np

# Optional imports with error handling
try:
    # MoviePy 2.x style import (direct from moviepy)
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy available")
except ImportError:
    try:
        # Try MoviePy 1.x style import as fallback
        from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
        MOVIEPY_AVAILABLE = True
        print("✅ MoviePy available (legacy import)")
    except ImportError:
        print("⚠️ MoviePy not available. Using FFmpeg-only processing.")
        VideoFileClip = None
        TextClip = None
        CompositeVideoClip = None
        MOVIEPY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    print("⚠️ PIL/Pillow not available. Caption generation may be limited.")
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️ OpenCV not available. Using basic video processing.")
    CV2_AVAILABLE = False

from .gemini_analyzer import EngagingSegment

class ClipGenerator:
    """Generates video clips from engaging segments"""
    
    def __init__(self):
        self.output_dir = os.getenv('OUTPUT_DIR', './output')
        self.temp_dir = os.getenv('TEMP_DIR', './temp')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Video settings
        self.target_width = int(os.getenv('OUTPUT_WIDTH', 1080))
        self.target_height = int(os.getenv('OUTPUT_HEIGHT', 1920))
        self.target_fps = int(os.getenv('OUTPUT_FPS', 30))
        
        # Caption settings
        self.caption_style = {
            'fontsize': 72,
            'font': 'Arial-Bold',
            'color': 'white',
            'stroke_color': 'black',
            'stroke_width': 4,
            'method': 'caption'
        }
    
    def create_clips(
        self, 
        video_path: str, 
        engaging_segments: List[EngagingSegment], 
        settings: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Create video clips from engaging segments
        
        Args:
            video_path: Path to source video
            engaging_segments: List of segments to clip
            settings: Processing settings
            
        Returns:
            List of created clip information
        """
        try:
            clips_info = []
            
            for i, segment in enumerate(engaging_segments):
                print(f"✂️ Creating clip {i+1}/{len(engaging_segments)}...")
                
                clip_info = self._create_single_clip(
                    video_path, segment, i+1, settings
                )
                
                if clip_info:
                    clips_info.append(clip_info)
                else:
                    print(f"❌ Failed to create clip {i+1}")
            
            print(f"✅ Successfully created {len(clips_info)} clips!")
            return clips_info
            
        except Exception as e:
            print(f"❌ Error creating clips: {e}")
            return []
    
    def _create_single_clip(
        self, 
        video_path: str, 
        segment: EngagingSegment, 
        clip_number: int, 
        settings: Dict[str, any]
    ) -> Optional[Dict[str, any]]:
        """Create a single video clip"""
        try:
            # Generate output filename
            safe_title = self._sanitize_filename(segment.suggested_title)
            output_filename = f"clip_{clip_number:02d}_{safe_title}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Extract and process clip
            success = self._extract_and_process_clip(
                video_path, segment, output_path, settings
            )
            
            if not success:
                return None
            
            # Get clip duration
            duration = segment.end_time - segment.start_time
            
            return {
                'clip_number': clip_number,
                'title': segment.suggested_title,
                'hook': segment.hook,
                'duration': round(duration, 1),
                'output_path': output_path,
                'hashtags': segment.hashtags,
                'keywords': segment.keywords,
                'segment_type': segment.segment_type,
                'engagement_score': round(segment.engagement_score, 2),
                'start_time': segment.start_time,
                'end_time': segment.end_time
            }
            
        except Exception as e:
            print(f"❌ Error creating clip {clip_number}: {e}")
            return None
    
    def _extract_and_process_clip(
        self, 
        video_path: str, 
        segment: EngagingSegment, 
        output_path: str, 
        settings: Dict[str, any]
    ) -> bool:
        """Extract clip and apply all processing"""
        try:
            # Always use FFmpeg as primary method (more reliable)
            if self._use_ffmpeg_processing(video_path, segment, output_path, settings):
                return True
                
            # Fallback to MoviePy if available and FFmpeg fails
            if MOVIEPY_AVAILABLE:
                return self._use_moviepy_processing(video_path, segment, output_path, settings)
            else:
                print("❌ Both FFmpeg and MoviePy processing failed")
                return False
            
        except Exception as e:
            print(f"❌ Error processing clip: {e}")
            return False
    
    def _use_moviepy_processing(
        self, 
        video_path: str, 
        segment: EngagingSegment, 
        output_path: str, 
        settings: Dict[str, any]
    ) -> bool:
        """Process clip using MoviePy (with captions)"""
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available")
            return False
            
        try:
            # Load video and extract segment
            with VideoFileClip(video_path) as video:
                # Extract the segment (MoviePy 2.x uses subclipped)
                clip = video.subclipped(segment.start_time, segment.end_time)
                
                # Resize to vertical format
                processed_clip = self._resize_to_vertical(clip)
                
                # Add captions if text is available
                if hasattr(segment, 'text') and segment.text:
                    processed_clip = self._add_captions_moviepy(processed_clip, segment.text)
                
                # Write the final clip
                processed_clip.write_videofile(
                    output_path,
                    fps=self.target_fps,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                processed_clip.close()
            
            return os.path.exists(output_path)
            
        except Exception as e:
            print(f"MoviePy processing failed: {e}")
            return False
    
    def _use_ffmpeg_processing(
        self,
        video_path: str,
        segment: EngagingSegment,
        output_path: str,
        settings: Dict[str, any]
    ) -> bool:
        """Process clip using FFmpeg with karaoke-style word-timed captions."""
        try:
            duration = segment.end_time - segment.start_time
            if duration <= 0:
                print(f"❌ Invalid segment duration: {duration}")
                return False

            cmd = [
                'ffmpeg',
                '-ss', str(segment.start_time),
                '-i', video_path,
                '-t', str(duration),
            ]

            # ----------------------------------------------------------------
            # Base video filters: scale to 9:16, blur-pad the background
            # ----------------------------------------------------------------
            vf_parts = [
                # Scale the content to fit inside 9:16, keeping aspect ratio
                f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease",
                # Pad the remaining space with a blurred copy of the video
                # ffmpeg's 'boxing' technique: overlay content on blurred background
                f"pad={self.target_width}:{self.target_height}:(ow-iw)/2:(oh-ih)/2:color=black",
            ]

            # ----------------------------------------------------------------
            # Caption overlay
            # ----------------------------------------------------------------
            caption_filters = self._build_caption_filters(segment, duration)
            vf_parts.extend(caption_filters)

            cmd.extend([
                '-vf', ','.join(vf_parts),
                '-r', str(self.target_fps),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path,
            ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return os.path.exists(output_path)
            else:
                print(f"FFmpeg error: {result.stderr[-800:]}")
                return False

        except Exception as e:
            print(f"FFmpeg processing failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Caption helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esc(text: str) -> str:
        """Escape a string for use inside an FFmpeg drawtext filter value."""
        return (
            text
            .replace('\\', '\\\\')
            .replace("'",  "\u2019")   # typographic right single quote (safe in drawtext)
            .replace(':',  '\\:')
            .replace('%',  '\\%')
            .replace('[',  '\\[')
            .replace(']',  '\\]')
            .replace('\n', ' ')
        )

    def _build_caption_filters(
        self,
        segment: EngagingSegment,
        clip_duration: float,
    ) -> List[str]:
        """
        Return a list of drawtext filter strings for the caption overlay.

        Strategy
        --------
        * If word-level timestamps are available (Whisper with word_timestamps=True),
          emit one drawtext entry per *group* of WORDS_PER_GROUP words, each
          shown only while those words are being spoken — karaoke style.
        * Otherwise fall back to a single static caption (first 12 words of the
          segment text) displayed for the whole clip duration.

        All captions are:
          - Large bold font (72 pt at 1080-wide)
          - Yellow text with a thick black stroke for maximum legibility
          - Horizontally centred, positioned at 78 % of the frame height
          - Wrapped at ~20 chars per line by drawtext's built-in :line_spacing
        """
        WORDS_PER_GROUP = 3   # words shown at once (matching viral TikTok/Reels style)
        FONT_SIZE       = 72
        STROKE_W        = 4
        Y_POS           = 'h*0.78'
        X_POS           = '(w-text_w)/2'
        TEXT_COLOR      = 'white'
        STROKE_COLOR    = 'black'
        BOX_COLOR       = '0x00000099'  # semi-transparent black
        BOX_BORDER      = 12            # px padding around text box

        def _dt(text: str, t_start: float, t_end: float) -> str:
            """Single drawtext filter string with timed enable expression."""
            escaped = self._esc(text)
            enable  = f"between(t,{t_start:.3f},{t_end:.3f})"
            return (
                f"drawtext=text='{escaped}'"
                f":fontsize={FONT_SIZE}"
                f":fontcolor={TEXT_COLOR}"
                f":borderw={STROKE_W}"
                f":bordercolor={STROKE_COLOR}"
                f":box=1:boxcolor={BOX_COLOR}:boxborderw={BOX_BORDER}"
                f":x={X_POS}:y={Y_POS}"
                f":line_spacing=6"
                f":fix_bounds=true"
                f":enable='{enable}'"
            )

        # ---- word-timed (karaoke) path ----
        words = segment.word_segments or []
        # Offset word timestamps to be relative to clip start
        clip_start = segment.start_time
        rel_words = [
            {'word': w['word'], 'start': w['start'] - clip_start, 'end': w['end'] - clip_start}
            for w in words
            if w.get('word')
        ]

        filters: List[str] = []

        if rel_words:
            # Group into chunks of WORDS_PER_GROUP
            for i in range(0, len(rel_words), WORDS_PER_GROUP):
                group = rel_words[i : i + WORDS_PER_GROUP]
                text     = ' '.join(w['word'] for w in group).upper()
                t_start  = max(0.0, group[0]['start'])
                t_end    = min(clip_duration, group[-1]['end'])
                if t_end <= t_start:
                    t_end = min(clip_duration, t_start + 1.0)
                filters.append(_dt(text, t_start, t_end))
        else:
            # ---- fallback: static caption for whole clip ----
            raw = (segment.text or '').strip()
            if not raw:
                return []
            # Limit to first 12 words and uppercase for readability
            words_fb = raw.split()[:12]
            # Split into two lines of 6 for narrow videos
            if len(words_fb) > 6:
                line1 = ' '.join(words_fb[:6])
                line2 = ' '.join(words_fb[6:])
                text  = f"{line1}\\n{line2}".upper()
            else:
                text = ' '.join(words_fb).upper()
            filters.append(_dt(text, 0.5, max(1.0, clip_duration - 0.5)))

        return filters
    
    def _resize_to_vertical(self, clip):
        """Resize video to vertical 9:16 format"""
        if not MOVIEPY_AVAILABLE:
            return clip  # Skip if MoviePy not available
            
        # Get original dimensions
        w, h = clip.size
        
        # Calculate target dimensions maintaining aspect ratio
        target_ratio = self.target_width / self.target_height
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Video is wider - crop sides (MoviePy 2.x uses cropped)
            new_width = int(h * target_ratio)
            x_center = w / 2
            x1 = int(x_center - new_width / 2)
            x2 = int(x_center + new_width / 2)
            clip = clip.cropped(x1=x1, x2=x2)
        else:
            # Video is taller - crop top/bottom
            new_height = int(w / target_ratio)
            y_center = h / 2
            y1 = int(y_center - new_height / 2)
            y2 = int(y_center + new_height / 2)
            clip = clip.cropped(y1=y1, y2=y2)
        
        # Resize to target resolution (MoviePy 2.x uses resized)
        return clip.resized((self.target_width, self.target_height))
    
    def _add_captions_moviepy(self, clip, text: str):
        """Add animated captions to clip using MoviePy"""
        if not MOVIEPY_AVAILABLE:
            print("⚠️ MoviePy not available for caption generation")
            return clip
            
        try:
            # Split text into manageable chunks
            words = text.split()
            chunks = self._split_text_into_chunks(words, max_words_per_chunk=8)
            
            if not chunks:
                return clip
            
            # Create caption clips
            caption_clips = []
            duration_per_chunk = clip.duration / len(chunks)
            
            for i, chunk in enumerate(chunks):
                start_time = i * duration_per_chunk
                end_time = (i + 1) * duration_per_chunk
                
                # Create text clip (MoviePy 2.x API)
                txt_clip = (
                    TextClip(
                        text=chunk,
                        font_size=self.caption_style['fontsize'],
                        font=self.caption_style['font'],
                        color=self.caption_style['color'],
                        stroke_color=self.caption_style['stroke_color'],
                        stroke_width=self.caption_style['stroke_width'],
                        method=self.caption_style['method'],
                        size=(self.target_width - 80, None),
                    )
                    .with_position(('center', 'bottom'))
                    .with_duration(duration_per_chunk)
                    .with_start(start_time)
                )
                
                caption_clips.append(txt_clip)
            
            # Composite video with captions
            return CompositeVideoClip([clip] + caption_clips)
            
        except Exception as e:
            print(f"Caption generation failed: {e}")
            return clip  # Return original clip if captions fail
    
    def _split_text_into_chunks(self, words: List[str], max_words_per_chunk: int = 8) -> List[str]:
        """Split text into readable chunks for captions"""
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            # Check if chunk is getting too long or ends with punctuation
            if (len(current_chunk) >= max_words_per_chunk or 
                word.endswith(('.', '!', '?'))):
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for file system"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        filename = filename[:50]
        
        # Remove extra whitespace and underscores
        filename = '_'.join(filename.split())
        
        return filename
    
    def _get_video_info_ffprobe(self, video_path: str) -> Dict[str, any]:
        """Get video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
            
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        return {}
    
    def create_preview_thumbnails(self, clips_info: List[Dict[str, any]]) -> List[str]:
        """Create preview thumbnails for clips"""
        thumbnail_paths = []
        
        for clip_info in clips_info:
            try:
                video_path = clip_info['output_path']
                thumbnail_path = video_path.replace('.mp4', '_thumb.jpg')
                
                # Extract frame at middle of clip using ffmpeg
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-ss', '50%',  # Middle of video
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    '-y',
                    thumbnail_path
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0:
                    thumbnail_paths.append(thumbnail_path)
                    clip_info['thumbnail_path'] = thumbnail_path
                
            except Exception as e:
                print(f"Failed to create thumbnail for {clip_info.get('title', 'clip')}: {e}")
        
        return thumbnail_paths
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")
    
    def optimize_clips_for_platform(self, clips_info: List[Dict[str, any]], platform: str = 'general'):
        """Optimize clips for specific social media platforms"""
        platform_settings = {
            'tiktok': {'max_size_mb': 287, 'max_duration': 60},
            'instagram': {'max_size_mb': 100, 'max_duration': 60},
            'youtube_shorts': {'max_size_mb': 256, 'max_duration': 60},
            'general': {'max_size_mb': 100, 'max_duration': 60}
        }
        
        settings = platform_settings.get(platform, platform_settings['general'])
        
        optimized_clips = []
        for clip_info in clips_info:
            try:
                video_path = clip_info['output_path']
                
                # Check file size
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                
                if file_size_mb > settings['max_size_mb']:
                    # Compress the video
                    compressed_path = video_path.replace('.mp4', f'_{platform}_optimized.mp4')
                    self._compress_video(video_path, compressed_path, settings['max_size_mb'])
                    clip_info['optimized_path'] = compressed_path
                
                optimized_clips.append(clip_info)
                
            except Exception as e:
                print(f"Error optimizing clip: {e}")
                optimized_clips.append(clip_info)  # Keep original
        
        return optimized_clips
    
    def _compress_video(self, input_path: str, output_path: str, target_size_mb: float):
        """Compress video to target file size"""
        try:
            # Get video duration for bitrate calculation
            info = self._get_video_info_ffprobe(input_path)
            duration = float(info.get('format', {}).get('duration', 60))
            
            # Calculate target bitrate (leaving room for audio)
            target_bitrate = int((target_size_mb * 8 * 1024 * 1024) / duration * 0.9)  # 90% for video
            
            cmd = [
                'ffmpeg', '-i', input_path,
                '-b:v', f'{target_bitrate}',
                '-maxrate', f'{target_bitrate}',
                '-bufsize', f'{target_bitrate * 2}',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True)
            
        except Exception as e:
            print(f"Video compression failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_temp_files()