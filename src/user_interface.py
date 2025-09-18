"""
Conversational User Interface for Video Editing Copilot

Handles natural language parsing, user interaction, and response generation.
"""

import re
from typing import Dict, List, Optional, Any
import random

class ConversationalInterface:
    """Handles conversational interaction with users"""
    
    def __init__(self):
        self.style_keywords = {
            'funny': ['funny', 'humor', 'comedy', 'laugh', 'hilarious', 'joke'],
            'educational': ['educational', 'learn', 'teach', 'explain', 'tutorial'],
            'energetic': ['energetic', 'exciting', 'hype', 'pump', 'intense'],
            'emotional': ['emotional', 'touching', 'heartfelt', 'moving', 'deep'],
            'viral': ['viral', 'trending', 'hook', 'attention', 'grabbing'],
            'professional': ['professional', 'business', 'corporate', 'clean']
        }
        
        # Enthusiastic responses
        self.processing_phrases = [
            "Perfect! Let me work my magic ✨",
            "Got it! Time to create something amazing 🎬",
            "Excellent choice! Processing your video now 🚀",
            "Love it! Let's turn this into viral content 📱"
        ]
        
        self.success_phrases = [
            "🎉 Your clips are ready! Here's what I created:",
            "✨ Mission accomplished! Check out these amazing clips:",
            "🔥 Done! Your social media content is ready to go:",
            "🎬 Perfect! Here are your fresh, engaging clips:"
        ]
    
    def parse_user_preferences(self, user_input: str) -> Dict[str, Any]:
        """
        Parse natural language input to extract user preferences
        
        Args:
            user_input: User's natural language request
            
        Returns:
            Dictionary of extracted preferences
        """
        preferences = {}
        input_lower = user_input.lower()
        
        # Extract clip count
        clip_patterns = [
            r'(\d+)\s+clips?',
            r'make\s+(\d+)',
            r'create\s+(\d+)',
            r'generate\s+(\d+)'
        ]
        
        for pattern in clip_patterns:
            match = re.search(pattern, input_lower)
            if match:
                preferences['clip_count'] = int(match.group(1))
                break
        
        # Extract clip length
        length_patterns = [
            r'(\d+)\s+seconds?',
            r'(\d+)s\b',
            r'(\d+)-(\d+)\s+seconds?'
        ]
        
        for pattern in length_patterns:
            match = re.search(pattern, input_lower)
            if match:
                if len(match.groups()) == 1:
                    # Single length specified
                    length = int(match.group(1))
                    preferences['clip_length_min'] = max(15, length - 5)
                    preferences['clip_length_max'] = min(120, length + 5)
                else:
                    # Range specified
                    preferences['clip_length_min'] = int(match.group(1))
                    preferences['clip_length_max'] = int(match.group(2))
                break
        
        # Extract style preferences
        detected_styles = []
        for style, keywords in self.style_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    detected_styles.append(style)
                    break
        
        if detected_styles:
            preferences['style'] = detected_styles[0]  # Use first detected style
        
        # Extract content type hints
        if any(word in input_lower for word in ['podcast', 'interview', 'talk']):
            preferences['content_type'] = 'conversation'
        elif any(word in input_lower for word in ['presentation', 'lecture', 'tutorial']):
            preferences['content_type'] = 'educational'
        elif any(word in input_lower for word in ['vlog', 'daily', 'lifestyle']):
            preferences['content_type'] = 'lifestyle'
        
        return preferences
    
    def request_video_input(self) -> Optional[str]:
        """Request video input from user"""
        print("\n📹 Please provide your video:")
        print("1. Drag and drop a video file")
        print("2. Paste a YouTube URL")
        print("3. Enter the path to your video file")
        
        video_input = input("Video source: ").strip()
        
        if not video_input:
            return None
            
        # Basic validation
        if video_input.startswith(('http://', 'https://')):
            return video_input
        elif any(video_input.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv']):
            return video_input
        else:
            # Assume it's a file path
            return video_input
    
    def generate_processing_start_response(self, settings: Dict[str, Any]) -> str:
        """Generate enthusiastic processing start response"""
        phrase = random.choice(self.processing_phrases)
        
        clip_count = settings.get('clip_count', 5)
        min_length = settings.get('clip_length_min', 30)
        max_length = settings.get('clip_length_max', 60)
        style = settings.get('style', 'engaging')
        
        length_desc = f"{min_length}-{max_length} seconds" if min_length != max_length else f"{min_length} seconds"
        
        details = f"\n📊 Here's the plan:\n"
        details += f"• Creating {clip_count} clips\n"
        details += f"• Length: {length_desc} each\n"
        details += f"• Style: {style.title()}\n"
        details += f"• Format: Vertical (9:16) with captions\n"
        
        return f"{phrase}{details}"
    
    def generate_success_response(self, clips: List[Dict[str, Any]], settings: Dict[str, Any]) -> str:
        """Generate success response with clip details"""
        phrase = random.choice(self.success_phrases)
        
        response = f"{phrase}\n\n"
        
        for i, clip in enumerate(clips, 1):
            response += f"📱 **Clip {i}**: {clip.get('title', f'Clip {i}')}\n"
            response += f"   • Duration: {clip.get('duration', 'Unknown')} seconds\n"
            response += f"   • Hook: \"{clip.get('hook', 'Engaging content')}\"\n"
            response += f"   • File: {clip.get('output_path', 'clip.mp4')}\n"
            
            if clip.get('hashtags'):
                response += f"   • Suggested tags: {', '.join(clip.get('hashtags', []))}\n"
            
            response += "\n"
        
        response += "💡 **Pro Tips:**\n"
        response += "• Post during peak hours (7-9 PM)\n"
        response += "• Use trending hashtags for your niche\n"
        response += "• Add a call-to-action in your caption\n"
        response += "• Cross-post to multiple platforms for maximum reach!\n"
        
        return response
    
    def generate_error_response(self, error_type: str, details: str = "") -> str:
        """Generate friendly error responses"""
        error_responses = {
            'no_video': "🤔 I need a video to work with! Please provide a video file or YouTube URL.",
            
            'transcription_failed': "😅 I'm having trouble hearing the audio clearly. Make sure your video has clear speech and try again.",
            
            'no_engaging_content': "🤷‍♂️ Hmm, I couldn't find clear engaging moments in this video. It might work better with content that has:\n• Clear speech\n• Conversations or dialogue\n• Topic changes or emotional moments\n• Q&A sections or storytelling",
            
            'clip_generation_failed': "⚠️ Something went wrong while creating your clips. This might be due to:\n• Video format issues\n• File corruption\n• Insufficient storage space\nTry with a different video or check your setup.",
            
            'processing_error': f"😬 Oops! Something unexpected happened during processing.\nError details: {details}\n\nDon't worry, let's try again! Make sure your video file is accessible and try a different approach."
        }
        
        base_response = error_responses.get(error_type, f"❌ An error occurred: {details}")
        
        # Add helpful suggestions
        suggestions = "\n\n💡 **Quick fixes to try:**\n"
        suggestions += "• Make sure your video has clear audio\n"
        suggestions += "• Try a shorter video (under 30 minutes works best)\n"
        suggestions += "• Check that the file isn't corrupted\n"
        suggestions += "• For YouTube links, make sure the video is public\n"
        
        return base_response + suggestions
    
    def generate_clarification_request(self, missing_info: str) -> str:
        """Generate requests for clarification"""
        clarification_requests = {
            'video_source': "📹 What video would you like me to work with? Share a file or YouTube URL!",
            'preferences': "🎯 Got it! Any specific preferences?\n• How many clips? (default: 5)\n• Length per clip? (default: 30-60 seconds)\n• Style focus? (funny, educational, energetic, etc.)",
            'style_clarification': "🎨 What vibe are you going for?\n• Funny/Comedy clips\n• Educational highlights\n• High-energy moments\n• Emotional/touching parts\n• Professional/business focused"
        }
        
        return clarification_requests.get(missing_info, f"Can you help me understand: {missing_info}?")