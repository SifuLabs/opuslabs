# Video Clipping Agent Specification

## Overview
Transform long-form videos into engaging short-form, social-ready clips optimized for TikTok, Instagram Reels, and YouTube Shorts.

## Core Workflow

### 1. Input Processing
- **Video Sources**: Accept file uploads or URL links (YouTube, social media)
- **User Preferences**: 
  - Clip length (default: 30-60 seconds)
  - Number of clips (default: 5)
  - Style preferences (energetic, educational, funny, etc.)
  - Custom prompts for content focus

### 2. Transcription Pipeline
- Use speech-to-text API (Whisper, AssemblyAI, or similar)
- Generate timestamped transcript with speaker diarization
- Clean and format transcript for analysis

### 3. Content Analysis & Selection
- **Hook Detection**: Identify compelling opening lines, questions, surprising statements
- **Emotional Analysis**: Find moments of high energy, laughter, surprise, or passion
- **Topic Shifts**: Detect natural conversation breaks and subject changes
- **Engagement Patterns**: Score segments based on:
  - Speech pace and energy
  - Keyword density (trending topics, power words)
  - Natural pause points for editing
  - Visual action (if available)

### 4. Clip Generation
- **Smart Segmentation**: Cut clips at natural conversation breaks
- **Buffer Management**: Add 2-3 seconds before/after for context
- **Quality Control**: Ensure each clip has clear beginning, middle, end

### 5. Video Processing
- **Format Conversion**: Transform to vertical 9:16 aspect ratio (1080x1920)
- **Cropping Strategy**: Focus on speakers, maintain visual interest
- **Caption Overlay**: 
  - Animated, readable captions
  - Highlight key phrases
  - Style options (modern, minimal, bold)

### 6. Output Delivery
- Generate preview thumbnails
- Provide download links for each clip
- Include metadata (suggested titles, hashtags, posting times)

## Technical Specifications

### Default Parameters
- **Clip Count**: 5 clips per video
- **Clip Length**: 30-60 seconds
- **Output Format**: MP4, 1080x1920, 30fps
- **Caption Style**: Modern animated with highlight effects

### Quality Requirements
- Maintain original audio quality
- Smooth transitions and cuts
- Readable captions on all devices
- Optimized file sizes for social platforms

### Supported Input Formats
- MP4, MOV, AVI, MKV
- YouTube URLs
- Social media links (when possible)

## Agent Behavior Guidelines

### Conversational Style
- Act as a creative assistant, not a technical API
- Use friendly, enthusiastic language
- Provide status updates during processing
- Ask clarifying questions when needed

### Error Handling
- Clear explanations when input is invalid
- Suggest alternatives when processing fails
- Graceful degradation (fewer clips if needed)

### Example Interactions

**User**: "Turn my podcast episode into clips."
**Agent**: "Perfect! I'll transcribe your podcast, find the most engaging moments, and create 5 TikTok-ready clips with captions. What's the main topic? Any specific moments you want me to focus on?"

**User**: "Make it funnier and 15 seconds each."
**Agent**: "Got it! I'll hunt for the funniest moments and keep clips short and punchy at 15 seconds. Processing now..."

## Performance Targets
- Transcription: < 2 minutes for 30-minute video
- Analysis: < 1 minute for transcript processing
- Video processing: < 5 minutes for 5 clips
- Total processing time: < 8 minutes end-to-end