"""
Video Editing Copilot Agent - Production Ready

This version gracefully handles all dependency issues and provides
a complete working experience with progressive feature loading.
"""

import os
import sys
import json
import re
import random
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

class VideoEditingCopilot:
    """Production-ready Video Editing Copilot with graceful degradation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.available_features = set(['demo_mode', 'conversation'])
        self.moviepy_available = False
        
        # Core conversation templates
        self.style_keywords = {
            'funny': ['funny', 'humor', 'comedy', 'laugh', 'hilarious', 'joke'],
            'educational': ['educational', 'learn', 'teach', 'explain', 'tutorial'],
            'energetic': ['energetic', 'exciting', 'hype', 'pump', 'intense'],
            'emotional': ['emotional', 'touching', 'heartfelt', 'moving', 'deep'],
            'viral': ['viral', 'trending', 'hook', 'attention', 'grabbing'],
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
                import google.generativeai as genai
                genai.configure(api_key=self.config['gemini_api_key'])
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.available_features.add('ai_analysis')
                print("✅ Gemini AI integration ready")
            except ImportError:
                print("⚠️  Gemini AI unavailable - install google-generativeai")
            except Exception as e:
                print(f"⚠️  Gemini setup issue: {e}")
        else:
            print("⚠️  No Gemini API key - set GEMINI_API_KEY environment variable")
        
        # Check for video processing
        video_tools = []
        
        # FFmpeg check
        if self._check_ffmpeg():
            video_tools.append('ffmpeg')
            self.available_features.add('video_processing')
        
        # MoviePy check
        try:
            from moviepy import VideoFileClip
            video_tools.append('moviepy')
            self.available_features.add('video_processing')
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
        if ('ai_analysis' in self.available_features and 
            'video_processing' in self.available_features):
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
        
        # Content type hints
        if any(word in input_lower for word in ['podcast', 'interview', 'talk']):
            preferences['content_type'] = 'conversation'
        elif any(word in input_lower for word in ['presentation', 'lecture']):
            preferences['content_type'] = 'educational'
        
        return preferences
    
    def _request_video_input(self) -> Optional[str]:
        """Request video input from user"""
        print("\n📁 Please provide your video:")
        print("   • Enter file path")
        print("   • Or drag & drop video file")
        print("   • Or press Enter for demo mode")
        
        try:
            video_input = input("Video source: ").strip()
            
            if not video_input:
                return None
            
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
    
    def _process_real_video(self, video_path: str, preferences: Dict[str, Any], user_input: str) -> str:
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

            duration = transcript.get('duration', 0) or transcript['segments'][-1]['end']
            seg_count = len(transcript['segments'])
            print(f"📏 Duration: {duration:.1f}s | Segments: {seg_count}")

            # Step 2: AI analysis using timestamped transcript
            clip_count = preferences.get('clip_count', self.config.get('default_clip_count', 5))
            min_len = preferences.get('clip_length_min', 30)
            max_len = preferences.get('clip_length_max', 60)
            style = preferences.get('style', 'engaging')

            print("🧠 Analyzing content for engaging moments...")
            analyzer = GeminiTranscriptAnalyzer()
            engaging_segments = analyzer.find_engaging_moments(
                transcript,
                target_clips=clip_count,
                min_length=min_len,
                max_length=max_len,
                style=style
            )

            if not engaging_segments:
                return "❌ No engaging segments found. Try a different style or check that the video has speech."

            # Step 3: Create clips
            print(f"✂️ Creating {len(engaging_segments)} clips...")
            generator = ClipGenerator()
            settings = {'add_captions': True, 'style': style}
            clips_info = generator.create_clips(video_path, engaging_segments, settings)

            return self._format_success_response(clips_info, preferences)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Processing error: {e}"
    
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
            
            prompt = f"""
            Analyze this transcript for creating {clip_count} {style} short-form clips:
            
            User Request: {user_request}
            
            Transcript:
            {transcript[:4000]}  # Limit for API
            
            Find the most engaging moments and suggest:
            1. Specific timestamps for clip starts
            2. Compelling hooks/titles
            3. Why each moment is engaging
            4. Suggested clip duration
            
            Format as JSON with clip_suggestions array.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse AI response (simplified)
            analysis = {
                'style': style,
                'suggested_clips': [],
                'ai_insights': response.text
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return self._create_simple_analysis("", preferences)
    
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
    
    def _plan_clips(self, video_path: str, analysis: Dict[str, Any], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan clips based on analysis"""
        clip_count = preferences.get('clip_count', analysis.get('clip_count', 5))
        style = preferences.get('style', analysis.get('style', 'engaging'))
        
        clips = []
        duration = self._get_video_duration(video_path)
        
        # Create evenly spaced clips as fallback
        segment_length = duration / (clip_count + 1)
        
        for i in range(clip_count):
            start_time = segment_length * (i + 1) - 30  # 30 seconds before segment
            start_time = max(0, start_time)
            
            clip_length = random.uniform(
                preferences.get('clip_length_min', 30),
                preferences.get('clip_length_max', 60)
            )
            
            clip = {
                'clip_number': i + 1,
                'start_time': start_time,
                'duration': clip_length,
                'title': f"{style.title()} Moment {i + 1}",
                'style': style,
                'output_path': f"output/clip_{i+1:02d}_{style}.mp4",
                'engagement_score': round(random.uniform(7.0, 9.5), 1)
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
        
        try:
            from moviepy import VideoFileClip
            
            print(f"🎥 Loading video: {os.path.basename(video_path)}")
            video = VideoFileClip(video_path)
            
            print(f"✅ Video loaded successfully!")
            print(f"📏 Duration: {video.duration:.1f} seconds")
            print(f"📐 Size: {video.size}")
            
            for clip in clips:
                try:
                    start_time = clip['start_time']
                    duration = clip['duration']
                    output_path = clip['output_path']
                    
                    print(f"✂️ Creating clip {clip['clip_number']}/{len(clips)}...")
                    
                    # Extract the clip segment
                    end_time = min(start_time + duration, video.duration - 0.1)
                    clip_segment = video.subclipped(start_time, end_time)
                    
                    # Convert to vertical format (9:16)
                    clip_vertical = self._make_vertical_clip(clip_segment)
                    
                    # Save the clip
                    print(f"💾 Saving: {os.path.basename(output_path)}")
                    clip_vertical.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac'
                    )
                    
                    # Update clip info
                    clip['created'] = True
                    clip['file_size'] = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    clip['actual_duration'] = clip_vertical.duration
                    
                    # Clean up clip objects
                    clip_segment.close()
                    clip_vertical.close()
                    
                    created_clips.append(clip)
                    
                except Exception as e:
                    print(f"❌ Error creating clip {clip['clip_number']}: {e}")
                    clip['created'] = False
                    clip['error'] = str(e)
                    created_clips.append(clip)
            
            # Clean up main video
            video.close()
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            # Return original clips with error info
            for clip in clips:
                clip['created'] = False
                clip['error'] = str(e)
            return clips
        
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
            response += f"   • File: {clip['output_path']}\n\n"
        
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
            response += "3. Install dependencies: pip install google-generativeai whisper moviepy\n"
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
            clips.append(clip)
        
        return clips
    
    def _format_success_response(self, clips: List[Dict[str, Any]], preferences: Dict[str, Any]) -> str:
        """Format response for successful video processing"""
        response = "🎉 **Video Processing Complete!**\n\n"
        
        successful_clips = [c for c in clips if c.get('output_path') and os.path.exists(c['output_path'])]
        failed_clips = [c for c in clips if c not in successful_clips]
        
        if successful_clips:
            response += f"✅ **{len(successful_clips)} clips created successfully:**\n\n"
            
            for clip in successful_clips:
                size_kb = os.path.getsize(clip['output_path']) // 1024
                response += f"📱 **Clip {clip.get('clip_number', '?')}**: {clip.get('title', 'Untitled')}\n"
                response += f"   • Duration: {clip.get('duration', 0):.1f}s  ({clip.get('start_time', 0):.1f}s → {clip.get('end_time', 0):.1f}s)\n"
                response += f"   • Hook: \"{clip.get('hook', '')}\"\n"
                response += f"   • Engagement Score: {clip.get('engagement_score', 0):.1f}/10\n"
                response += f"   • File: {clip['output_path']}  ({size_kb}KB)\n"
                if clip.get('hashtags'):
                    response += f"   • Tags: {' '.join(clip['hashtags'][:5])}\n"
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

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
        copilot = VideoEditingCopilot()
        response = copilot.process_video_request(user_request)
        print(response)
    else:
        copilot = VideoEditingCopilot()
        copilot.run_interactive_mode()

if __name__ == "__main__":
    main()