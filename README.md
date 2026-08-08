# 🎬 Video Editing Copilot Agent

**Transform your long-form videos into viral short-form clips using AI!**

An intelligent video editing assistant that analyzes your content and automatically creates engaging clips optimized for TikTok, Instagram Reels, and YouTube Shorts.

## ✨ What It Does

- 🎤 **Analyzes video content** using AI transcription and Gemini analysis
- 🧠 **Finds engaging moments** automatically using advanced algorithms  
- ✂️ **Creates optimized clips** in vertical 9:16 format with captions
- 🎯 **Tailors to your style** - funny, educational, viral, professional
- 📈 **Packages clips for growth** with publish titles, captions, hashtags, CTAs, thumbnail prompts, and posting windows
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

User: "Make 5 YouTube Shorts for subscribers about fitness."
Agent: "I'll create Shorts and package each one with a publish title, caption, tags, thumbnail text, comment prompt, CTA, and recommended posting window."

User: "Use smart crop with minimal captions at the top, caption color #ffaa00, caption size 64, and watermark \"Opus Labs\"."
Agent: "I'll track the visible subject and apply those caption and brand controls."

User: "Use a split-screen conversation layout with logo \"assets/mark.png\" and subtitles in Spanish and Swahili."
Agent: "I'll create the two-speaker layout, add the image logo, and export translated SRT sidecars."
```

## Editing and brand controls

- Reframing: `smart crop`, `blur background`, `center crop`, `show full frame`,
  or `split-screen conversation layout`.
- Captions: bold, clean, or minimal themes; top, middle, or bottom safe
  positions; custom colors and sizes.
- Branding: text watermarks, PNG/JPG/WebP logos, and named kits such as
  `save brand kit "Studio" ...` followed by `use brand kit "Studio"`.
- Corrections: use `replace "Open Eye" with "OpenAI"`, or provide
  `transcript corrections "corrections.json"` with a JSON string mapping or a
  `replacements` list containing `from` and `to` fields.
- Translation: request up to five languages with `subtitles in Spanish and
  Swahili`. Translation requires `google-genai` plus `GEMINI_API_KEY` (or
  `GOOGLE_API_KEY`); source-language SRT export still works without it.

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
- Publish-ready title variants, captions, hashtags, and SEO tags
- Thumbnail text and thumbnail generation prompts
- Posting windows, CTAs, and engagement questions
- Preview thumbnails
- Source and optional translated SRT subtitle files
