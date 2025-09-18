"""
Video Processing Pipeline

Handles video input, transcription, and preprocessing for the clipping agent.
"""

import os
import tempfile
import yt_dlp
from typing import Dict, List, Optional, Tuple
import subprocess
import json
from pathlib import Path

# Optional imports with error handling
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Whisper import failed: {e}")
    print("   Speech recognition will use API-based services only")
    whisper = None
    WHISPER_AVAILABLE = False

class VideoProcessor:
    """Handles video processing and transcription"""
    
    def __init__(self):
        self.temp_dir = os.getenv('TEMP_DIR', './temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize Whisper model (you can also use OpenAI API or AssemblyAI)
        self.whisper_model = None
        self.use_local_whisper = WHISPER_AVAILABLE  # Only use if available
    
    def _load_whisper_model(self):
        """Lazy load Whisper model"""
        if not WHISPER_AVAILABLE:
            raise Exception("Whisper is not available. Please use API-based transcription.")
            
        if self.whisper_model is None:
            print("Loading speech recognition model...")
            self.whisper_model = whisper.load_model("base")
    
    def download_video(self, url: str) -> Optional[str]:
        """
        Download video from URL (YouTube, etc.)
        
        Args:
            url: Video URL
            
        Returns:
            Path to downloaded video file or None if failed
        """
        try:
            output_path = os.path.join(self.temp_dir, "downloaded_video.%(ext)s")
            
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[height<=720]',  # Limit quality for faster processing
                'extractaudio': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'Unknown')
                
                # Find the actual downloaded file
                for file in os.listdir(self.temp_dir):
                    if file.startswith("downloaded_video."):
                        downloaded_path = os.path.join(self.temp_dir, file)
                        print(f"✅ Downloaded: {video_title}")
                        return downloaded_path
                        
            return None
            
        except Exception as e:
            print(f"❌ Failed to download video: {e}")
            return None
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate that the video file exists and is accessible
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(video_path):
            return False
        
        try:
            # Quick validation using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                # Check if it has video stream
                has_video = any(stream.get('codec_type') == 'video' for stream in info.get('streams', []))
                return has_video
            
            return False
            
        except Exception:
            # Fallback: just check file existence and extension
            return any(video_path.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm'])
    
    def get_video_info(self, video_path: str) -> Dict[str, any]:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Extract relevant info
                video_stream = next((stream for stream in info['streams'] if stream['codec_type'] == 'video'), None)
                audio_stream = next((stream for stream in info['streams'] if stream['codec_type'] == 'audio'), None)
                
                return {
                    'duration': float(info['format'].get('duration', 0)),
                    'width': video_stream.get('width') if video_stream else None,
                    'height': video_stream.get('height') if video_stream else None,
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else None,
                    'has_audio': audio_stream is not None,
                    'filesize': int(info['format'].get('size', 0))
                }
        except Exception as e:
            print(f"Warning: Could not get video info: {e}")
            
        return {'duration': 0, 'width': None, 'height': None, 'fps': None, 'has_audio': True, 'filesize': 0}
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video for transcription
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        try:
            audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
            
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(audio_path):
                return audio_path
            else:
                print(f"❌ Audio extraction failed")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None
    
    def transcribe_video(self, video_path: str) -> Optional[Dict[str, any]]:
        """
        Transcribe video to text with timestamps
        
        Args:
            video_path: Path to video file or URL
            
        Returns:
            Transcript data with timestamps or None if failed
        """
        try:
            # Handle URL downloads
            if video_path.startswith(('http://', 'https://')):
                video_path = self.download_video(video_path)
                if not video_path:
                    return None
            
            # Validate video file
            if not self.validate_video_file(video_path):
                print(f"❌ Invalid video file: {video_path}")
                return None
            
            # Get video info
            video_info = self.get_video_info(video_path)
            print(f"📊 Video duration: {video_info['duration']:.1f} seconds")
            
            if self.use_local_whisper:
                return self._transcribe_with_whisper(video_path)
            else:
                return self._transcribe_with_gemini_api(video_path)
                
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def _transcribe_with_whisper(self, video_path: str) -> Optional[Dict[str, any]]:
        """Transcribe using local Whisper model"""
        if not WHISPER_AVAILABLE:
            print("❌ Whisper not available, falling back to Gemini API")
            return self._transcribe_with_gemini_api(video_path)
            
        try:
            self._load_whisper_model()
            
            print("🎤 Transcribing audio (this may take a few minutes)...")
            result = self.whisper_model.transcribe(video_path, verbose=False)
            
            # Format transcript data
            transcript_data = {
                'text': result['text'],
                'segments': [
                    {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'confidence': getattr(segment, 'confidence', 0.9)  # Whisper doesn't always provide confidence
                    }
                    for segment in result['segments']
                ],
                'language': result.get('language', 'en'),
                'duration': result['segments'][-1]['end'] if result['segments'] else 0
            }
            
            print(f"✅ Transcription complete! Found {len(transcript_data['segments'])} segments")
            return transcript_data
            
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            print("🔄 Falling back to Gemini API...")
            return self._transcribe_with_gemini_api(video_path)
    
    def _transcribe_with_gemini_api(self, video_path: str) -> Optional[Dict[str, any]]:
        """Transcribe using Gemini API for enhanced analysis"""
        try:
            import google.generativeai as genai
        except ImportError:
            print("❌ google-generativeai not installed. Please install with: pip install google-generativeai")
            return self._create_simple_transcript(video_path)
        
        try:
            # Extract audio first
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                return None
            
            print("🎤 Transcribing with Gemini API...")
            
            # Configure Gemini
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                print("❌ GOOGLE_API_KEY not found in environment")
                return self._create_simple_transcript(video_path)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Upload audio file
            audio_file = genai.upload_file(audio_path)
            
            # Create prompt for transcription with timestamps
            prompt = """
            Please transcribe this audio file with the following format:
            - Provide accurate timestamps for each segment
            - Include natural speech patterns and pauses
            - Identify speaker changes if multiple speakers
            - Format as JSON with segments containing: start_time, end_time, text, speaker (if applicable)
            - Ensure timestamps are in seconds as decimal numbers
            
            Focus on accuracy and natural speech flow for video clipping purposes.
            """
            
            response = model.generate_content([prompt, audio_file])
            
            # Parse Gemini response (it might not be perfect JSON, so we'll handle it)
            try:
                # Try to extract JSON from response
                import json
                import re
                
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    gemini_data = json.loads(json_match.group())
                else:
                    # Fallback: parse the text response manually
                    return self._parse_gemini_text_response(response.text, audio_path)
                
                # Format for our system
                if 'segments' in gemini_data:
                    transcript_data = {
                        'text': ' '.join([seg['text'] for seg in gemini_data['segments']]),
                        'segments': [
                            {
                                'start': float(seg.get('start_time', 0)),
                                'end': float(seg.get('end_time', 0)),
                                'text': seg.get('text', '').strip(),
                                'confidence': 0.9  # Gemini doesn't provide confidence scores
                            }
                            for seg in gemini_data['segments']
                        ],
                        'language': 'en',
                        'duration': gemini_data['segments'][-1].get('end_time', 0) if gemini_data['segments'] else 0
                    }
                else:
                    # Fallback to simple text parsing
                    return self._parse_gemini_text_response(response.text, audio_path)
                
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, use text parsing fallback
                return self._parse_gemini_text_response(response.text, audio_path)
            
            print(f"✅ Gemini transcription complete! Found {len(transcript_data['segments'])} segments")
            return transcript_data
            
        except Exception as e:
            print(f"❌ Gemini transcription failed: {e}")
            print("🔄 Using simple fallback transcription...")
            return self._create_simple_transcript(video_path)
    
    def _create_simple_transcript(self, video_path: str) -> Optional[Dict[str, any]]:
        """Create a simple transcript placeholder when all else fails"""
        try:
            # Get video duration
            video_info = self.get_video_info(video_path)
            duration = video_info.get('duration', 60)
            
            # Create simple segments every 10 seconds
            segments = []
            for i in range(0, int(duration), 10):
                start_time = i
                end_time = min(i + 10, duration)
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': f"Audio segment from {start_time}s to {end_time}s",
                    'confidence': 0.5
                })
            
            return {
                'text': f"Audio content from {os.path.basename(video_path)} ({duration:.1f} seconds)",
                'segments': segments,
                'language': 'en',
                'duration': duration
            }
            
        except Exception as e:
            print(f"❌ Simple transcript creation failed: {e}")
            return None
    
    def _parse_gemini_text_response(self, text_response: str, audio_path: str) -> Optional[Dict[str, any]]:
        """Parse Gemini's text response when JSON parsing fails"""
        try:
            # Get audio duration using ffprobe instead of librosa
            video_info = self.get_video_info(audio_path)
            duration = video_info.get('duration', 60)
            
            # Split text into sentences for rough segmentation
            sentences = [s.strip() for s in text_response.split('.') if s.strip()]
            
            if not sentences:
                return None
            
            # Create rough time segments
            segment_duration = duration / len(sentences)
            segments = []
            
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': sentence.strip(),
                    'confidence': 0.8
                })
            
            return {
                'text': text_response,
                'segments': segments,
                'language': 'en',
                'duration': duration
            }
            
        except Exception as e:
            print(f"❌ Gemini text parsing failed: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_temp_files()