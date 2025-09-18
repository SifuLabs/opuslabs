# 🎬 Video Editing Copilot Agent

**Transform your long-form videos into viral short-form clips using AI!**

An intelligent video editing assistant that analyzes your content and automatically creates engaging clips optimized for TikTok, Instagram Reels, and YouTube Shorts.

## ✨ What It Does

- 🎤 **Analyzes video content** using AI transcription and Gemini analysis
- 🧠 **Finds engaging moments** automatically using advanced algorithms  
- ✂️ **Creates optimized clips** in vertical 9:16 format with captions
- 🎯 **Tailors to your style** - funny, educational, viral, professional
- 💬 **Conversational interface** - just tell it what you want!
- **Conversational Interface**: Natural language interaction for preferences and customization
- **Cost-Effective**: Uses Google Gemini API for affordable AI processing

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your Google Gemini API key and other settings
```

3. Run the agent:
```bash
python main.py
```

## Usage Examples

```
User: "Turn my podcast episode into clips."
Agent: "Perfect! I'll transcribe your podcast, find the most engaging moments, and create 5 TikTok-ready clips with captions. What's the main topic?"

User: "Make 3 clips, 45 seconds each, focus on the funny parts."
Agent: "Got it! Hunting for the funniest moments and creating 45-second clips. Processing now..."
```

## Configuration

Default settings:
- 5 clips per video
- 30-60 seconds per clip
- 1080x1920 vertical format
- Animated captions

All settings are customizable through the conversational interface.

## Supported Input Formats

- Video files: MP4, MOV, AVI, MKV
- YouTube URLs
- Direct video links

## Output

- MP4 files optimized for social media
- Animated captions overlay
- Suggested titles and hashtags
- Preview thumbnails"# opuslabs" 
