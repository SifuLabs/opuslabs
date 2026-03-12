"""
Enhanced Transcript Analysis with Gemini AI

Uses Google's Gemini API for intelligent content analysis and moment detection.
Includes: disk caching, transcript compression, and retry with exponential backoff
to minimise API credit usage and handle rate limits gracefully.
"""

import os
import re
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Optional imports with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ google-generativeai not available. Install with: pip install google-generativeai")
    genai = None
    GEMINI_AVAILABLE = False

try:
    import nltk
    from textstat import flesch_reading_ease
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️ NLTK or textstat not available. Install with: pip install nltk textstat")
    nltk = None
    NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration (all tunable via environment variables)
# ---------------------------------------------------------------------------

# Max characters of compressed transcript sent to Gemini per request.
# ~4 000 chars ≈ ~1 000 tokens — enough for good clip selection while
# keeping costs low. Raise to 8000 on a paid plan.
MAX_TRANSCRIPT_CHARS = int(os.getenv('GEMINI_MAX_TRANSCRIPT_CHARS', '4000'))

# Gemini model. gemini-1.5-flash-8b is the free-tier-friendly default
# (higher RPM, lower cost). Switch to gemini-1.5-flash for more capability.
DEFAULT_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-8b')

# Cache directory. Set GEMINI_CACHE_DIR=off to disable caching entirely.
_cache_dir_env = os.getenv('GEMINI_CACHE_DIR', './.cache/gemini')
CACHE_DIR = Path(_cache_dir_env) if _cache_dir_env.lower() != 'off' else None

@dataclass
class EngagingSegment:
    """Represents an engaging video segment"""
    start_time: float
    end_time: float
    text: str
    hook: str
    engagement_score: float
    segment_type: str  # 'hook', 'climax', 'story', 'question', 'reveal'
    keywords: List[str]
    suggested_title: str
    hashtags: List[str]
    emotion: str  # 'funny', 'exciting', 'educational', 'surprising'
    viral_potential: float
    # Per-word timing for karaoke-style captions (populated when Whisper
    # word_timestamps=True was used; empty list otherwise)
    word_segments: List[Dict] = None

    def __post_init__(self):
        if self.word_segments is None:
            self.word_segments = []

class GeminiTranscriptAnalyzer:
    """Enhanced transcript analysis using Gemini AI"""
    
    def __init__(self):
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY (prefer GEMINI_API_KEY)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(DEFAULT_MODEL)
            self.use_ai_analysis = True
            print(f"✅ Gemini model: {DEFAULT_MODEL}")
        else:
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini not available. Using rule-based analysis only.")
            elif not api_key:
                print("⚠️ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
            self.use_ai_analysis = False

        # Set up disk cache
        self._cache_enabled = CACHE_DIR is not None
        if self._cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download NLTK data if needed, with graceful fallback
        if NLTK_AVAILABLE:
            # Newer NLTK uses punkt_tab; older versions use punkt.
            for resource in ('tokenizers/punkt_tab', 'tokenizers/punkt'):
                try:
                    nltk.data.find(resource)
                    break
                except LookupError:
                    pass
            else:
                # Neither found — try to download both silently
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)

            try:
                from nltk.tokenize import sent_tokenize, word_tokenize
                # Smoke-test so we catch missing data early
                word_tokenize('test')
                self.sent_tokenize = sent_tokenize
                self.word_tokenize = word_tokenize
            except Exception:
                # NLTK data still missing — fall back to regex tokenisers
                NLTK_AVAILABLE_local = False
                self.sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
                self.word_tokenize = lambda text: re.findall(r"[a-zA-Z']+", text)
        else:
            # Simple fallback tokenizers
            self.sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
            self.word_tokenize = lambda text: re.findall(r"[a-zA-Z']+", text)
    
    def find_engaging_moments(
        self, 
        transcript: Dict[str, any], 
        target_clips: int = 5,
        min_length: int = 30,
        max_length: int = 60,
        style: str = 'engaging'
    ) -> List[EngagingSegment]:
        """
        Find engaging moments using AI-powered analysis
        """
        try:
            segments = transcript.get('segments', [])
            if not segments:
                return []
            
            full_text = transcript.get('text', '')
            
            if self.use_ai_analysis and full_text:
                # Use Gemini for intelligent analysis
                return self._analyze_with_gemini(
                    segments, full_text, target_clips, min_length, max_length, style
                )
            else:
                # Fallback to rule-based analysis
                return self._analyze_with_rules(
                    segments, target_clips, min_length, max_length, style
                )
                
        except Exception as e:
            print(f"❌ Error in transcript analysis: {e}")
            # Fallback to simple rule-based analysis
            return self._analyze_with_rules(
                transcript.get('segments', []), target_clips, min_length, max_length, style
            )
    
    def _analyze_with_gemini(
        self, 
        segments: List[Dict], 
        full_text: str,
        target_clips: int,
        min_length: int,
        max_length: int,
        style: str
    ) -> List[EngagingSegment]:
        """Use Gemini AI — with disk caching and retry on rate limits"""

        # 1. Check disk cache first — free, instant, no API call
        cache_key = self._cache_key(segments, target_clips, min_length, max_length, style)
        cached = self._load_cache(cache_key)
        if cached is not None:
            result = self._create_segments_from_ai_analysis(segments, cached, style)
            if result:
                return result

        print("🧠 Analyzing content with Gemini AI...")

        # 2. Build compressed transcript (fewer tokens = lower cost + less rate limiting)
        timestamped_transcript = self._build_timestamped_transcript(segments)
        prompt = self._create_analysis_prompt(
            timestamped_transcript, target_clips, min_length, max_length, style
        )

        # 3. Call Gemini with exponential backoff on 429 / quota errors
        ai_analysis = self._call_gemini_with_retry(prompt)

        if not ai_analysis:
            print("⚠️ AI analysis returned no data, falling back to rule-based")
            return self._analyze_with_rules(segments, target_clips, min_length, max_length, style)

        # 4. Persist result so next run costs nothing
        self._save_cache(cache_key, ai_analysis)

        result = self._create_segments_from_ai_analysis(segments, ai_analysis, style)
        if not result:
            print("⚠️ No valid segments from AI, falling back to rule-based")
            return self._analyze_with_rules(segments, target_clips, min_length, max_length, style)
        return result

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """Call Gemini with exponential backoff on rate-limit (429) errors."""
        delay = 30  # seconds before first retry
        for attempt in range(1, max_retries + 2):  # +1 for the initial attempt
            try:
                response = self.model.generate_content(prompt)
                return self._parse_gemini_analysis(response.text)
            except Exception as e:
                err = str(e)
                is_rate_limit = '429' in err or 'RATE_LIMIT' in err or 'Quota exceeded' in err
                if is_rate_limit and attempt <= max_retries:
                    print(f"⏳ Rate limit hit — waiting {delay}s before retry {attempt}/{max_retries}...")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)  # cap at 2 minutes
                    continue
                # Non-retriable error or retries exhausted
                print(f"❌ Gemini call failed: {e}")
                if is_rate_limit:
                    print("   Tip: the result is cached after a successful run, so retries won't cost extra.")
                    print(f"   Or set GEMINI_MODEL=gemini-2.0-flash-lite in your .env for higher free-tier limits.")
                    print(f"   Current model: {DEFAULT_MODEL}")
                return None
        return None

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------
    def _cache_key(self, segments: List[Dict], target_clips: int,
                   min_length: int, max_length: int, style: str) -> str:
        """Stable hash over all analysis inputs."""
        payload = json.dumps({
            'segs': [(s.get('start'), s.get('end'), s.get('text', '')) for s in segments],
            'n': target_clips, 'min': min_length, 'max': max_length,
            'style': style, 'model': DEFAULT_MODEL,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[Dict]:
        if not self._cache_enabled:
            return None
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                print(f"⚡ Loaded cached analysis ({key[:8]}…) — no API call needed")
                return data
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, data: Dict) -> None:
        if not self._cache_enabled:
            return
        try:
            (CACHE_DIR / f"{key}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(f"💾 Analysis cached ({key[:8]}…) — future runs won't call the API")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Transcript compression
    # -----------------------------------------------------------------------
    def _build_timestamped_transcript(self, segments: List[Dict]) -> str:
        """
        Build a *compressed* timestamped transcript to minimise token usage.

        Strategy:
          1. Merge consecutive short segments (<8 words) into bigger ones.
          2. If the result still exceeds MAX_TRANSCRIPT_CHARS, sample evenly
             so Gemini sees the whole video but at lower resolution.
        """
        # Step 1 — merge short consecutive segments
        merged: List[Dict] = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue
            if merged and len(merged[-1]['text'].split()) < 8:
                merged[-1]['text'] += ' ' + text
                merged[-1]['end'] = seg.get('end', merged[-1]['end'])
            else:
                merged.append({'start': seg['start'],
                                'end': seg.get('end', seg['start']),
                                'text': text})

        lines = [f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" for s in merged]

        # Step 2 — if still too long, sample evenly across the video
        full = '\n'.join(lines)
        if len(full) <= MAX_TRANSCRIPT_CHARS:
            return full

        total = len(lines)
        head_n = max(1, int(total * 0.15))
        tail_n = max(1, int(total * 0.10))
        mid_lines = lines[head_n:-tail_n]

        # How many middle lines fit in the remaining budget?
        budget_for_mid = MAX_TRANSCRIPT_CHARS - sum(len(l) + 1 for l in lines[:head_n]) \
                         - sum(len(l) + 1 for l in lines[-tail_n:]) - 60
        if budget_for_mid > 0 and mid_lines:
            avg_line = max(1, sum(len(l) for l in mid_lines) // len(mid_lines))
            keep_n = max(1, budget_for_mid // avg_line)
            step = max(1, len(mid_lines) // keep_n)
            mid_lines = mid_lines[::step]

        compressed_lines = (
            lines[:head_n]
            + [f"… [{total - head_n - tail_n} segments condensed] …"]
            + mid_lines
            + lines[-tail_n:]
        )
        compressed = '\n'.join(compressed_lines)
        saved_pct = int((1 - len(compressed) / len(full)) * 100)
        print(f"📉 Transcript compressed by ~{saved_pct}% ({len(full)} → {len(compressed)} chars)")
        return compressed
    
    def _create_analysis_prompt(
        self, 
        timestamped_transcript: str, 
        target_clips: int, 
        min_length: int, 
        max_length: int, 
        style: str
    ) -> str:
        """Create a detailed prompt for Gemini using the timestamped transcript"""
        
        style_instructions = {
            'funny': 'Focus on humorous moments, jokes, funny stories, and comedic timing.',
            'educational': 'Identify key learning points, explanations, insights, and "aha" moments.',
            'energetic': 'Find high-energy moments, excitement, passion, and motivational content.',
            'emotional': 'Locate emotionally compelling moments, personal stories, and touching content.',
            'viral': 'Identify content with viral potential: surprising facts, bold statements, conflicts.',
            'professional': 'Find business insights, professional advice, and valuable takeaways.',
            'engaging': 'Look for any highly engaging content that would capture attention.'
        }
        
        style_instruction = style_instructions.get(style, style_instructions['engaging'])
        
        return f"""You are an expert social media video editor. Analyze the timestamped transcript below and identify the {target_clips} best moments for short-form clips.

CONTENT STYLE: {style_instruction}
CLIP LENGTH: Each clip must be {min_length}-{max_length} seconds long.

TIMESTAMPED TRANSCRIPT (format: [start_seconds - end_seconds] text):
{timestamped_transcript}

INSTRUCTIONS:
- Use the exact second values from the timestamps in the transcript above.
- Each clip must start and end at timestamps that appear in the transcript.
- Clips must not overlap each other.
- Prefer clips that start with a strong hook and end on a satisfying note.
- Choose moments with high engagement potential for social media.

OUTPUT: Respond with ONLY valid JSON (no markdown, no explanation). Use this exact schema:
{{
    "engaging_moments": [
        {{
            "title": "Punchy social-media title (max 10 words)",
            "hook": "The most compelling sentence from this clip",
            "start_time": 12.50,
            "end_time": 54.30,
            "engagement_score": 8.5,
            "viral_potential": 7.2,
            "emotion": "funny|exciting|educational|surprising|inspiring|emotional",
            "segment_type": "hook|story|reveal|climax|question",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "hashtags": ["#tag1", "#tag2", "#tag3"],
            "reason": "One sentence explaining why this moment is engaging"
        }}
    ]
}}

IMPORTANT:
- start_time and end_time MUST be exact float values from the transcript timestamps.
- Provide exactly {target_clips} moments with no overlaps.
- Score engagement_score and viral_potential from 1.0 to 10.0.
"""
    
    def _parse_gemini_analysis(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse Gemini's JSON analysis response"""
        try:
            # Strip markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
                cleaned = cleaned.strip()
            
            # Try direct JSON parse first
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            # Look for JSON object in the response
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to parse Gemini response: {e}")
            return None
    
    def _create_segments_from_ai_analysis(
        self, 
        segments: List[Dict], 
        ai_analysis: Dict[str, Any],
        style: str
    ) -> List[EngagingSegment]:
        """Convert AI analysis to EngagingSegment objects using direct timestamps"""
        
        engaging_moments = ai_analysis.get('engaging_moments', [])
        if not engaging_moments:
            return []
        
        max_video_time = segments[-1]['end'] if segments else 0
        result_segments = []
        
        for moment in engaging_moments:
            # Use direct timestamps from Gemini response
            start_time = moment.get('start_time')
            end_time = moment.get('end_time')
            
            if start_time is None or end_time is None:
                print(f"⚠️ Skipping moment '{moment.get('title', '?')}' — missing timestamps")
                continue
            
            try:
                start_time = float(start_time)
                end_time = float(end_time)
            except (TypeError, ValueError):
                print(f"⚠️ Skipping moment '{moment.get('title', '?')}' — invalid timestamps")
                continue
            
            # Snap timestamps to nearest real segment boundary for accuracy
            start_time = self._snap_to_segment(segments, start_time, 'start')
            end_time = self._snap_to_segment(segments, end_time, 'end')
            
            # Clamp to video bounds
            start_time = max(0.0, min(start_time, max_video_time))
            end_time = max(start_time + 5.0, min(end_time, max_video_time))
            
            segment_text = self._extract_text_for_timerange(segments, start_time, end_time)
            word_segs   = self._extract_words_for_timerange(segments, start_time, end_time)

            engaging_segment = EngagingSegment(
                start_time=start_time,
                end_time=end_time,
                text=segment_text,
                hook=moment.get('hook', 'Engaging content'),
                engagement_score=float(moment.get('engagement_score', 7.0)),
                segment_type=moment.get('segment_type', 'hook'),
                keywords=moment.get('keywords', []),
                suggested_title=moment.get('title', f'Clip {len(result_segments)+1}'),
                hashtags=moment.get('hashtags', ['#viral', '#shorts']),
                emotion=moment.get('emotion', 'engaging'),
                viral_potential=float(moment.get('viral_potential', 6.0)),
                word_segments=word_segs,
            )
            
            result_segments.append(engaging_segment)
        
        return result_segments

    def _snap_to_segment(self, segments: List[Dict], time: float, boundary: str) -> float:
        """Snap a time value to the nearest real segment start or end boundary"""
        if not segments:
            return time
        
        key = 'start' if boundary == 'start' else 'end'
        closest = min(segments, key=lambda s: abs(s[key] - time))
        
        # Only snap if within 3 seconds of a real boundary
        if abs(closest[key] - time) <= 3.0:
            return closest[key]
        return time
    
    def _extract_text_for_timerange(self, segments: List[Dict], start_time: float, end_time: float) -> str:
        """Extract text for a specific time range"""
        texts = []
        for segment in segments:
            if segment['end'] >= start_time and segment['start'] <= end_time:
                texts.append(segment['text'].strip())
        return ' '.join(texts)

    def _extract_words_for_timerange(
        self, segments: List[Dict], start_time: float, end_time: float
    ) -> List[Dict]:
        """Collect per-word timing entries that fall within the clip time range."""
        words: List[Dict] = []
        for seg in segments:
            if seg['end'] < start_time or seg['start'] > end_time:
                continue
            for w in seg.get('words', []):
                w_start = float(w.get('start', seg['start']))
                w_end   = float(w.get('end',   seg['end']))
                if w_end >= start_time and w_start <= end_time:
                    words.append({
                        'word':  w.get('word', '').strip(),
                        'start': w_start,
                        'end':   w_end,
                    })
        return words
    
    def _analyze_with_rules(
        self,
        segments: List[Dict],
        target_clips: int,
        min_length: int,
        max_length: int,
        style: str
    ) -> List[EngagingSegment]:
        """Fallback rule-based analysis"""

        print("🔧 Using rule-based content analysis...")

        if not segments:
            print("⚠️  No transcript segments available.")
            return []

        # ------------------------------------------------------------------ #
        # Score every segment                                                  #
        # ------------------------------------------------------------------ #
        excitement_words = {
            'amazing', 'incredible', 'wow', 'unbelievable', 'shocking',
            'never', 'always', 'everyone', 'secret', 'truth', 'actually',
            'important', 'key', 'critical', 'massive', 'huge',
        }
        style_words = {
            'funny':       {'funny', 'laugh', 'joke', 'hilarious', 'lol'},
            'educational': {'learn', 'understand', 'because', 'explain', 'means', 'shows'},
            'energetic':   {'incredible', 'amazing', 'lets', 'go', 'power', 'strong'},
            'emotional':   {'feel', 'heart', 'love', 'miss', 'proud', 'tears', 'story'},
            'viral':       {'secret', 'truth', 'nobody', 'everyone', 'never', 'always'},
        }

        seg_scores: List[Dict] = []
        for i, seg in enumerate(segments):
            text = seg.get('text', '').lower()
            score = 0.0
            score += text.count('?') * 2
            score += sum(2 for w in excitement_words if w in text)
            for w in style_words.get(style, set()):
                if w in text:
                    score += 3
            wc = len(text.split())
            if 15 <= wc <= 60:
                score += 1
            seg_scores.append({'index': i, 'score': score, 'text': seg.get('text', '')})

        seg_scores.sort(key=lambda x: x['score'], reverse=True)

        # ------------------------------------------------------------------ #
        # Build clips by expanding a window around the best-scored segment    #
        # until the clip reaches min_length (capped at max_length).           #
        # ------------------------------------------------------------------ #
        total_video_duration = segments[-1]['end'] - segments[0]['start']
        # Relax min_length if the whole video is shorter
        effective_min = min(min_length, total_video_duration * 0.5)

        result_segments: List[EngagingSegment] = []
        used_indices: set = set()

        # Try every scored candidate (not just top target_clips*2) so we fill
        # the requested clip count even for low-scored or short videos.
        for scored in seg_scores:
            if len(result_segments) >= target_clips:
                break
            center = scored['index']
            if center in used_indices:
                continue

            # Expand left and right until we reach effective_min or max_length
            lo, hi = center, center
            while True:
                dur = segments[hi]['end'] - segments[lo]['start']
                if dur >= effective_min:
                    break
                # Prefer expanding toward whichever end gives more time
                can_left  = lo > 0 and all(i not in used_indices for i in range(lo - 1, lo))
                can_right = hi < len(segments) - 1 and all(i not in used_indices for i in range(hi + 1, hi + 2))
                if not can_left and not can_right:
                    break
                left_gain  = (segments[lo]['start'] - segments[lo - 1]['start']) if can_left  else 0
                right_gain = (segments[hi + 1]['end'] - segments[hi]['end'])      if can_right else 0
                if can_right and right_gain >= left_gain:
                    hi += 1
                elif can_left:
                    lo -= 1
                else:
                    hi += 1

            # Trim from the right if we went over max_length
            while hi > lo and (segments[hi]['end'] - segments[lo]['start']) > max_length:
                hi -= 1

            start_time = segments[lo]['start']
            end_time   = segments[hi]['end']
            duration   = end_time - start_time

            # Accept the clip if it's at least half of effective_min
            if duration < effective_min * 0.5:
                continue

            # Avoid clips that heavily overlap a previously chosen clip
            overlap = any(
                not (end_time <= segments[segments.index(
                        next((s for s in segments if s['start'] == rs.start_time), segments[0]))
                    ]['start'] or
                    start_time >= rs.end_time)
                for rs in result_segments
            )
            # Simpler non-overlap check via used_indices
            window_indices = set(range(lo, hi + 1))
            if window_indices & used_indices:
                continue

            for idx in window_indices:
                used_indices.add(idx)

            clip_text  = ' '.join(segments[idx]['text'] for idx in range(lo, hi + 1))
            clip_words = self._extract_words_for_timerange(segments, start_time, end_time)
            hook_text  = scored['text']
            result_segments.append(EngagingSegment(
                start_time=start_time,
                end_time=end_time,
                text=clip_text,
                hook=(hook_text[:100] + '…') if len(hook_text) > 100 else hook_text,
                engagement_score=min(max(scored['score'], 1.0), 10.0),
                segment_type='hook',
                keywords=self._extract_simple_keywords(clip_text),
                suggested_title=f"Engaging Moment {len(result_segments) + 1}",
                hashtags=['#viral', '#shorts', '#content'],
                emotion=style,
                viral_potential=min(max(scored['score'] * 0.8, 1.0), 10.0),
                word_segments=clip_words,
            ))

        print(f"✅ Found {len(result_segments)} engaging moments using rule-based analysis")
        return result_segments
    
    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        words = self.word_tokenize(text.lower())
        
        # Filter common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        keywords = [word for word in words if len(word) > 3 and word not in common_words and word.isalpha()]
        
        # Return top 5 most frequent
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:5]

# Create an alias for backward compatibility
TranscriptAnalyzer = GeminiTranscriptAnalyzer