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

## Persistent projects and batch jobs

Create a project with one or more local videos or URLs. Each source receives a
separate isolated workspace and queued job:

```bash
python main.py project create "August launch" "episode-1.mp4" "episode-2.mp4"
python main.py project list
python main.py project add-source PROJECT_ID "episode-3.mp4"
python main.py job enqueue PROJECT_ID "Create 3 clean YouTube Shorts with smart crop"
python main.py job run --project PROJECT_ID
python main.py job list --project PROJECT_ID
```

Jobs persist progress and an event timeline in SQLite. Manage interrupted or
failed work with:

```bash
python main.py job cancel JOB_ID
python main.py job retry JOB_ID
python main.py job recover --stale-seconds 3600
python main.py job events JOB_ID
```

Multiple `job run` processes may safely share the queue: claims are atomic and
outputs live under `.opuslabs/workspaces/PROJECT_ID/JOB_ID/`. Transcript and
clip-analysis checkpoints are reused after recovery. Use `job recover --force`
only when you know no worker is still processing the persisted running jobs.

## Approval-gated publishing drafts

Publishing is deliberately two-step. First approve the exact clip bytes, then
submit that approval to a provider:

```bash
python main.py publish approve "output/clip.mp4" --by "reviewer@example.com"
python main.py publish submit "output/clip.mp4" \
  --approval APPROVAL_ID \
  --provider local \
  --platform youtube_shorts \
  --mode draft \
  --title "A useful short" \
  --caption "Watch this useful moment." \
  --hashtag Shorts \
  --privacy private
python main.py publish list
python main.py publish providers
```

Approvals contain a SHA-256 fingerprint. Editing or replacing a clip after
approval invalidates it. The built-in `local` provider writes an auditable JSON
draft under `.opuslabs/publishing/drafts/` and explicitly records that no
external upload occurred.

Live providers plug into the same draft/direct/schedule contract. OAuth uses
state validation and PKCE; token bundles never enter SQLite and are protected by
Windows DPAPI or an installed OS keyring. No live OAuth adapters ship by default
because each platform requires its own developer application, scopes, callback
URL, and review. Check availability with:

```bash
python main.py account oauth-providers
python main.py account list
python main.py account disconnect ACCOUNT_ID
```

## Analytics feedback loop

Import actual clip outcomes from a UTF-8 JSON or CSV export. Imports are
idempotent, so importing the same file again does not double-count results:

```bash
python main.py analytics schema
python main.py analytics import "analytics-export.json"
python main.py analytics report
python main.py analytics report --platform youtube_shorts --style educational
python main.py analytics list --platform youtube_shorts --limit 25
```

JSON accepts an array or a `records` array. This publication-linked example is
enriched with the approved clip path, fingerprint, provider, and platform:

```json
{
  "records": [
    {
      "publication_id": "PUBLICATION_ID",
      "style": "educational",
      "segment_type": "explanation",
      "keywords": ["workflow", "editing"],
      "predicted_engagement_score": 7.8,
      "views": 12500,
      "engaged_views": 8100,
      "retention_percent": 64.5,
      "shares": 410,
      "conversions": 95,
      "observed_at": "2026-08-08T12:00:00Z"
    }
  ]
}
```

Rows may instead use `clip_sha256` or `clip_path`. To reuse generated metadata,
provide `manifest_path` plus `clip_number`; the importer fills the platform,
style, prediction, segment type, keywords, and clip link from
`clip_manifest.json`. CSV uses the same field names. Run `analytics schema` for
the complete contract.

Reports compare predicted engagement with a transparent 0-10 observed score
derived from retention, engaged-view rate, shares, and conversions. Each
style/platform recommendation includes its observed sample size and rates.
When history exists for future processing, the analyzer requests a wider
candidate pool and reranks it using confidence-shrunk outcomes for style,
segment type, and keywords. Original AI prediction values are preserved, and
queued jobs save the reranking evidence in their checkpoints.

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
