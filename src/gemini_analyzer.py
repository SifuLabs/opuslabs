"""
Enhanced Transcript Analysis with Gemini AI

Uses Google's Gemini API for intelligent content analysis and moment detection.
"""

import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

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

class GeminiTranscriptAnalyzer:
    """Enhanced transcript analysis using Gemini AI"""
    
    def __init__(self):
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY (prefer GEMINI_API_KEY)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.use_ai_analysis = True
        else:
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini not available. Using rule-based analysis only.")
            elif not api_key:
                print("⚠️ No Gemini API key found. Set GEMINI_API_KEY environment variable.")
            self.use_ai_analysis = False
        
        # Download NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            from nltk.tokenize import sent_tokenize, word_tokenize
            self.sent_tokenize = sent_tokenize
            self.word_tokenize = word_tokenize
        else:
            # Simple fallback tokenizers
            self.sent_tokenize = lambda text: text.split('. ')
            self.word_tokenize = lambda text: text.split()
    
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
        """Use Gemini AI for intelligent content analysis with timestamped transcript"""
        try:
            print("🧠 Analyzing content with Gemini AI...")
            
            # Build a timestamped transcript string so Gemini can return real timestamps
            timestamped_transcript = self._build_timestamped_transcript(segments)
            
            # Create analysis prompt that includes actual timestamps
            prompt = self._create_analysis_prompt(
                timestamped_transcript, target_clips, min_length, max_length, style
            )
            
            # Get AI analysis
            response = self.model.generate_content(prompt)
            ai_analysis = self._parse_gemini_analysis(response.text)
            
            if not ai_analysis:
                print("⚠️ AI analysis failed, falling back to rule-based")
                return self._analyze_with_rules(segments, target_clips, min_length, max_length, style)
            
            # Convert AI analysis to segments using direct timestamps
            result = self._create_segments_from_ai_analysis(segments, ai_analysis, style)
            
            if not result:
                print("⚠️ No valid segments from AI, falling back to rule-based")
                return self._analyze_with_rules(segments, target_clips, min_length, max_length, style)
            
            return result
            
        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            print("🔄 Falling back to rule-based analysis...")
            return self._analyze_with_rules(segments, target_clips, min_length, max_length, style)

    def _build_timestamped_transcript(self, segments: List[Dict]) -> str:
        """Build a transcript string with timestamps that Gemini can reference"""
        lines = []
        for seg in segments:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            if text:
                lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
        return '\n'.join(lines)
    
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
                viral_potential=float(moment.get('viral_potential', 6.0))
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
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment overlaps with our range
            if seg_end >= start_time and seg_start <= end_time:
                texts.append(segment['text'].strip())
        
        return ' '.join(texts)
    
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
        
        # Simple rule-based scoring
        scored_segments = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].lower()
            
            # Basic engagement scoring
            score = 0
            
            # Question bonus
            if '?' in text:
                score += 2
            
            # Excitement words
            excitement_words = ['amazing', 'incredible', 'wow', 'unbelievable', 'shocking']
            score += sum(2 for word in excitement_words if word in text)
            
            # Style-specific scoring
            if style == 'funny' and any(word in text for word in ['funny', 'laugh', 'joke', 'hilarious']):
                score += 3
            elif style == 'educational' and any(word in text for word in ['learn', 'understand', 'because']):
                score += 2
            
            # Length bonus (prefer medium length)
            word_count = len(text.split())
            if 20 <= word_count <= 50:
                score += 1
            
            scored_segments.append({
                'index': i,
                'score': score,
                'segment': segment,
                'text': segment['text']
            })
        
        # Sort by score and select top segments
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        
        # Create engaging segments
        result_segments = []
        used_indices = set()
        
        for scored in scored_segments[:target_clips * 2]:  # Get more options
            if scored['index'] in used_indices:
                continue
            
            # Create a clip around this segment
            start_idx = max(0, scored['index'] - 1)
            end_idx = min(len(segments) - 1, scored['index'] + 3)
            
            start_time = segments[start_idx]['start']
            end_time = segments[end_idx]['end']
            duration = end_time - start_time
            
            # Check duration constraints
            if min_length <= duration <= max_length:
                # Mark indices as used
                for idx in range(start_idx, end_idx + 1):
                    used_indices.add(idx)
                
                # Extract text
                clip_text = ' '.join([segments[idx]['text'] for idx in range(start_idx, end_idx + 1)])
                
                segment = EngagingSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=clip_text,
                    hook=scored['text'][:100] + "..." if len(scored['text']) > 100 else scored['text'],
                    engagement_score=min(scored['score'], 10),
                    segment_type='hook',
                    keywords=self._extract_simple_keywords(clip_text),
                    suggested_title=f"Engaging Moment {len(result_segments) + 1}",
                    hashtags=['#viral', '#shorts', '#content'],
                    emotion=style,
                    viral_potential=min(scored['score'] * 0.8, 10)
                )
                
                result_segments.append(segment)
                
                if len(result_segments) >= target_clips:
                    break
        
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