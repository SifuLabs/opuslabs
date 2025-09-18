"""
Configuration and Utility Functions

Shared utilities and configuration for the Video Editing Copilot.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration management"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment and config files"""
        # Default configuration
        self.config = {
            # API Keys
            'google_api_key': os.getenv('GOOGLE_API_KEY'),
            'assemblyai_api_key': os.getenv('ASSEMBLYAI_API_KEY'),
            
            # Video Processing
            'default_clip_count': int(os.getenv('DEFAULT_CLIP_COUNT', 5)),
            'default_clip_length_min': int(os.getenv('DEFAULT_CLIP_LENGTH_MIN', 30)),
            'default_clip_length_max': int(os.getenv('DEFAULT_CLIP_LENGTH_MAX', 60)),
            'output_width': int(os.getenv('OUTPUT_WIDTH', 1080)),
            'output_height': int(os.getenv('OUTPUT_HEIGHT', 1920)),
            'output_fps': int(os.getenv('OUTPUT_FPS', 30)),
            
            # Paths
            'temp_dir': os.getenv('TEMP_DIR', './temp'),
            'output_dir': os.getenv('OUTPUT_DIR', './output'),
            'models_dir': os.getenv('MODELS_DIR', './models'),
            
            # Processing Options
            'use_local_whisper': True,
            'whisper_model': 'base',  # tiny, base, small, medium, large
            'max_video_duration': 1800,  # 30 minutes
            'max_file_size_mb': 500,
            
            # Caption Settings
            'caption_fontsize': 48,
            'caption_font': 'Arial-Bold',
            'caption_color': 'white',
            'caption_stroke_color': 'black',
            'caption_stroke_width': 3
        }
        
        # Load from config file if exists
        config_file = Path('./config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def save_config(self, filename: str = 'config.json'):
        """Save current configuration to file"""
        try:
            config_to_save = {k: v for k, v in self.config.items() 
                            if not k.endswith('_api_key')}  # Don't save API keys
            
            with open(filename, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

# Global config instance
config = Config()

def setup_directories():
    """Create necessary directories"""
    directories = [
        config.get('temp_dir'),
        config.get('output_dir'), 
        config.get('models_dir')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    dependencies = {}
    
    # Check ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        dependencies['ffmpeg'] = result.returncode == 0
    except:
        dependencies['ffmpeg'] = False
    
    # Check Python packages
    required_packages = [
        'whisper', 'moviepy', 'opencv-python', 'nltk', 
        'textstat', 'google-generativeai', 'yt-dlp', 'pillow'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"

def get_video_platforms() -> Dict[str, Dict[str, Any]]:
    """Get video platform specifications"""
    return {
        'tiktok': {
            'name': 'TikTok',
            'aspect_ratio': '9:16',
            'resolution': '1080x1920',
            'max_duration': 60,
            'max_file_size_mb': 287,
            'recommended_fps': 30
        },
        'instagram': {
            'name': 'Instagram Reels',
            'aspect_ratio': '9:16', 
            'resolution': '1080x1920',
            'max_duration': 60,
            'max_file_size_mb': 100,
            'recommended_fps': 30
        },
        'youtube_shorts': {
            'name': 'YouTube Shorts',
            'aspect_ratio': '9:16',
            'resolution': '1080x1920', 
            'max_duration': 60,
            'max_file_size_mb': 256,
            'recommended_fps': 30
        },
        'twitter': {
            'name': 'Twitter',
            'aspect_ratio': '16:9',
            'resolution': '1280x720',
            'max_duration': 140,
            'max_file_size_mb': 512,
            'recommended_fps': 30
        }
    }

class ProgressTracker:
    """Track processing progress"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.status_message = "Starting..."
    
    def update(self, step: int, message: str = ""):
        """Update progress"""
        self.current_step = min(step, self.total_steps)
        if message:
            self.status_message = message
        
        # Simple progress display
        progress = (self.current_step / self.total_steps) * 100
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r[{bar}] {progress:.1f}% - {self.status_message}", end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
    
    def complete(self, message: str = "Complete!"):
        """Mark as complete"""
        self.update(self.total_steps, message)

def print_system_info():
    """Print system information and dependencies"""
    print("🎬 Video Editing Copilot Agent - System Check")
    print("=" * 50)
    
    # Check dependencies
    deps = validate_dependencies()
    
    print("\n📋 Dependencies:")
    for dep, status in deps.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {dep}")
    
    # Check directories
    print(f"\n📁 Directories:")
    print(f"• Temp: {config.get('temp_dir')}")
    print(f"• Output: {config.get('output_dir')}")
    print(f"• Models: {config.get('models_dir')}")
    
    # Configuration summary
    print(f"\n⚙️ Configuration:")
    print(f"• Default clips: {config.get('default_clip_count')}")
    print(f"• Clip length: {config.get('default_clip_length_min')}-{config.get('default_clip_length_max')}s")
    print(f"• Output format: {config.get('output_width')}x{config.get('output_height')} @ {config.get('output_fps')}fps")
    print(f"• Whisper model: {config.get('whisper_model')} ({'local' if config.get('use_local_whisper') else 'API'})")
    
    print("\n🚀 Ready to create amazing content!")

if __name__ == "__main__":
    setup_directories()
    print_system_info()