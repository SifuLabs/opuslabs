"""
Transcript Analysis Module

AI-powered analysis to find engaging moments, hooks, and optimal clip segments.
"""

import re
import os
from typing import Dict, List, Optional, Tuple, Set
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import openai
from dataclasses import dataclass
import json

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

class TranscriptAnalyzer:
    """Analyzes transcripts to find engaging moments"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        self.stop_words = set(stopwords.words('english'))
        self.sent_tokenize = sent_tokenize
        self.word_tokenize = word_tokenize
        
        # Engagement indicators
        self.hook_patterns = [
            r'^(what if|imagine|here\'s the thing|listen|wait|stop|pause)',
            r'(shocking|incredible|amazing|unbelievable|mind-blowing)',
            r'(secret|truth|reality|fact|problem)',
            r'(never|always|everyone|nobody|everything)',
            r'\?$',  # Questions
            r'(you won\'t believe|this will|this is why)',
        ]
        
        self.energy_words = {
            'high': ['amazing', 'incredible', 'fantastic', 'awesome', 'wow', 'omg', 'crazy', 'insane', 'unreal'],
            'emotional': ['love', 'hate', 'fear', 'excited', 'angry', 'sad', 'happy', 'shocked', 'surprised'],
            'action': ['run', 'jump', 'fight', 'chase', 'grab', 'throw', 'break', 'smash', 'crash'],
            'urgency': ['now', 'immediately', 'urgent', 'quick', 'fast', 'hurry', 'rush', 'emergency']
        }
        
        self.conversation_markers = [
            'so', 'but', 'however', 'actually', 'basically', 'essentially', 'fundamentally',
            'here\'s the thing', 'the point is', 'what i mean is', 'in other words'
        ]
        
    def find_engaging_moments(
        self, 
        transcript: Dict[str, any], 
        target_clips: int = 5,
        min_length: int = 30,
        max_length: int = 60,
        style: str = 'engaging'
    ) -> List[EngagingSegment]:
        """
        Find the most engaging moments in the transcript
        
        Args:
            transcript: Transcript data with segments and timestamps
            target_clips: Number of clips to create
            min_length: Minimum clip length in seconds
            max_length: Maximum clip length in seconds
            style: Style preference (funny, educational, energetic, etc.)
            
        Returns:
            List of engaging segments for clipping
        """
        try:
            segments = transcript.get('segments', [])
            if not segments:
                return []
            
            # Analyze each segment for engagement
            analyzed_segments = self._analyze_segments(segments, style)
            
            # Find optimal clip boundaries
            potential_clips = self._find_clip_boundaries(
                analyzed_segments, min_length, max_length
            )
            
            # Score and rank clips
            scored_clips = self._score_clips(potential_clips, style)
            
            # Select best clips avoiding overlap
            selected_clips = self._select_non_overlapping_clips(scored_clips, target_clips)
            
            # Generate metadata for each clip
            final_clips = self._generate_clip_metadata(selected_clips, style)
            
            print(f"✅ Found {len(final_clips)} engaging moments")
            return final_clips
            
        except Exception as e:
            print(f"❌ Error analyzing transcript: {e}")
            return []
    
    def _analyze_segments(self, segments: List[Dict], style: str) -> List[Dict]:
        """Analyze individual segments for engagement indicators"""
        analyzed = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if len(text) < 10:  # Skip very short segments
                continue
            
            analysis = {
                'index': i,
                'start': segment['start'],
                'end': segment['end'],
                'text': text,
                'word_count': len(self.word_tokenize(text)),
                'sentence_count': len(self.sent_tokenize(text)),
                'hooks': self._detect_hooks(text),
                'energy_level': self._calculate_energy_level(text),
                'question_count': text.count('?'),
                'exclamation_count': text.count('!'),
                'readability': self._calculate_readability(text),
                'keywords': self._extract_keywords(text),
                'emotional_weight': self._calculate_emotional_weight(text, style),
                'conversation_flow': self._analyze_conversation_flow(text, i, segments)
            }
            
            analyzed.append(analysis)
        
        return analyzed
    
    def _detect_hooks(self, text: str) -> List[str]:
        """Detect hook patterns in text"""
        hooks = []
        text_lower = text.lower()
        
        for pattern in self.hook_patterns:
            matches = re.findall(pattern, text_lower)
            hooks.extend(matches)
        
        return hooks
    
    def _calculate_energy_level(self, text: str) -> float:
        """Calculate energy level based on word choice and punctuation"""
        text_lower = text.lower()
        words = self.word_tokenize(text_lower)
        
        energy_score = 0
        total_words = len(words)
        
        if total_words == 0:
            return 0
        
        # Count energy words
        for category, word_list in self.energy_words.items():
            matches = sum(1 for word in words if word in word_list)
            if category == 'high':
                energy_score += matches * 3
            elif category == 'emotional':
                energy_score += matches * 2
            elif category == 'action':
                energy_score += matches * 2.5
            elif category == 'urgency':
                energy_score += matches * 2
        
        # Add punctuation bonus
        energy_score += text.count('!') * 1.5
        energy_score += text.count('?') * 1
        energy_score += len(re.findall(r'[A-Z]{2,}', text)) * 1  # All caps words
        
        return min(energy_score / total_words * 10, 10)  # Normalize to 0-10
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            if len(text) < 20:
                return 5  # Default for short text
            
            # Flesch Reading Ease (higher = easier to read)
            ease_score = flesch_reading_ease(text)
            return min(max(ease_score / 10, 0), 10)  # Normalize to 0-10
        except:
            return 5  # Default
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = self.word_tokenize(text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words 
            and len(word) > 3 
            and word.isalpha()
        ]
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:5]]
    
    def _calculate_emotional_weight(self, text: str, style: str) -> float:
        """Calculate emotional weight based on style preference"""
        text_lower = text.lower()
        
        if style == 'funny':
            funny_words = ['funny', 'laugh', 'hilarious', 'joke', 'comedy', 'haha', 'lol']
            score = sum(2 for word in funny_words if word in text_lower)
            
        elif style == 'educational':
            edu_words = ['learn', 'understand', 'explain', 'because', 'therefore', 'reason']
            score = sum(1.5 for word in edu_words if word in text_lower)
            
        elif style == 'energetic':
            return self._calculate_energy_level(text)
            
        else:  # General engagement
            engaging_words = ['amazing', 'incredible', 'important', 'secret', 'truth', 'problem']
            score = sum(1 for word in engaging_words if word in text_lower)
        
        return min(score, 10)
    
    def _analyze_conversation_flow(self, text: str, index: int, segments: List[Dict]) -> float:
        """Analyze conversation flow and natural breaks"""
        text_lower = text.lower()
        
        flow_score = 0
        
        # Check for conversation markers
        for marker in self.conversation_markers:
            if marker in text_lower:
                flow_score += 1
        
        # Check for topic transitions
        if any(phrase in text_lower for phrase in ['moving on', 'next', 'another thing', 'also']):
            flow_score += 2
        
        # Check for story elements
        if any(word in text_lower for word in ['story', 'happened', 'then', 'suddenly']):
            flow_score += 1.5
        
        return min(flow_score, 5)
    
    def _find_clip_boundaries(
        self, 
        analyzed_segments: List[Dict], 
        min_length: int, 
        max_length: int
    ) -> List[Dict]:
        """Find optimal clip boundaries"""
        potential_clips = []
        
        for i in range(len(analyzed_segments)):
            start_segment = analyzed_segments[i]
            
            # Try different end points
            for j in range(i, len(analyzed_segments)):
                end_segment = analyzed_segments[j]
                
                duration = end_segment['end'] - start_segment['start']
                
                # Check if duration is within bounds
                if min_length <= duration <= max_length:
                    # Combine text from all segments in this clip
                    clip_text = ' '.join([
                        seg['text'] for seg in analyzed_segments[i:j+1]
                    ])
                    
                    potential_clips.append({
                        'start_time': start_segment['start'],
                        'end_time': end_segment['end'],
                        'duration': duration,
                        'text': clip_text,
                        'segments': analyzed_segments[i:j+1],
                        'start_index': i,
                        'end_index': j
                    })
        
        return potential_clips
    
    def _score_clips(self, potential_clips: List[Dict], style: str) -> List[Dict]:
        """Score clips based on engagement factors"""
        scored_clips = []
        
        for clip in potential_clips:
            segments = clip['segments']
            
            # Calculate composite score
            total_energy = sum(seg['energy_level'] for seg in segments)
            total_hooks = sum(len(seg['hooks']) for seg in segments)
            total_questions = sum(seg['question_count'] for seg in segments)
            total_emotional = sum(seg['emotional_weight'] for seg in segments)
            avg_readability = sum(seg['readability'] for seg in segments) / len(segments)
            
            # Weighted scoring
            engagement_score = (
                total_energy * 0.3 +
                total_hooks * 2.0 +
                total_questions * 1.5 +
                total_emotional * 0.4 +
                avg_readability * 0.2
            )
            
            # Bonus for good length
            ideal_length = 45  # seconds
            length_bonus = 1 - abs(clip['duration'] - ideal_length) / ideal_length * 0.3
            engagement_score *= max(length_bonus, 0.7)
            
            clip['engagement_score'] = engagement_score
            scored_clips.append(clip)
        
        # Sort by score
        return sorted(scored_clips, key=lambda x: x['engagement_score'], reverse=True)
    
    def _select_non_overlapping_clips(self, scored_clips: List[Dict], target_count: int) -> List[Dict]:
        """Select non-overlapping clips with highest scores"""
        selected = []
        used_segments = set()
        
        for clip in scored_clips:
            # Check for overlap with already selected clips
            clip_segments = set(range(clip['start_index'], clip['end_index'] + 1))
            
            if not clip_segments.intersection(used_segments):
                selected.append(clip)
                used_segments.update(clip_segments)
                
                if len(selected) >= target_count:
                    break
        
        return selected
    
    def _generate_clip_metadata(self, clips: List[Dict], style: str) -> List[EngagingSegment]:
        """Generate metadata for each clip"""
        engaging_segments = []
        
        for i, clip in enumerate(clips):
            # Extract hook (first compelling phrase)
            hook = self._extract_best_hook(clip['text'])
            
            # Generate title
            title = self._generate_title(clip['text'], style, i + 1)
            
            # Generate hashtags
            hashtags = self._generate_hashtags(clip['text'], style)
            
            # Determine segment type
            segment_type = self._classify_segment_type(clip['text'])
            
            # Extract keywords
            keywords = self._extract_keywords(clip['text'])
            
            segment = EngagingSegment(
                start_time=clip['start_time'],
                end_time=clip['end_time'],
                text=clip['text'],
                hook=hook,
                engagement_score=clip['engagement_score'],
                segment_type=segment_type,
                keywords=keywords,
                suggested_title=title,
                hashtags=hashtags
            )
            
            engaging_segments.append(segment)
        
        return engaging_segments
    
    def _extract_best_hook(self, text: str) -> str:
        """Extract the most compelling hook from text"""
        sentences = self.sent_tokenize(text)
        
        if not sentences:
            return text[:50] + "..." if len(text) > 50 else text
        
        # Look for questions first
        for sentence in sentences[:3]:  # Check first 3 sentences
            if '?' in sentence:
                return sentence.strip()
        
        # Look for hook patterns
        for sentence in sentences[:2]:  # Check first 2 sentences
            sentence_lower = sentence.lower()
            for pattern in self.hook_patterns:
                if re.search(pattern, sentence_lower):
                    return sentence.strip()
        
        # Default to first sentence
        return sentences[0].strip()
    
    def _generate_title(self, text: str, style: str, clip_number: int) -> str:
        """Generate engaging title for clip"""
        # Extract key phrases
        words = self.word_tokenize(text.lower())
        important_words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        if not important_words:
            return f"Clip {clip_number}"
        
        # Style-specific title generation
        if style == 'funny':
            prefixes = ["This is Hilarious", "You'll Laugh At This", "Comedy Gold"]
        elif style == 'educational':
            prefixes = ["Learn This", "Key Insight", "Important Point"]
        elif style == 'energetic':
            prefixes = ["High Energy Moment", "This is Intense", "Amazing Content"]
        else:
            prefixes = ["Must Watch", "Engaging Moment", "Key Highlight"]
        
        # Combine with key terms
        key_term = important_words[0].title() if important_words else ""
        prefix = prefixes[clip_number % len(prefixes)]
        
        if key_term:
            return f"{prefix}: {key_term}"
        else:
            return prefix
    
    def _generate_hashtags(self, text: str, style: str) -> List[str]:
        """Generate relevant hashtags"""
        base_tags = ['#shorts', '#viral', '#content']
        
        # Style-specific tags
        style_tags = {
            'funny': ['#comedy', '#funny', '#laugh'],
            'educational': ['#learn', '#knowledge', '#education'],
            'energetic': ['#energy', '#motivation', '#hype'],
            'professional': ['#business', '#professional', '#tips']
        }
        
        tags = base_tags + style_tags.get(style, ['#engaging'])
        
        # Add content-based tags
        keywords = self._extract_keywords(text)
        for keyword in keywords[:2]:  # Add top 2 keywords as hashtags
            tag = f"#{keyword.lower()}"
            if tag not in tags:
                tags.append(tag)
        
        return tags[:8]  # Limit to 8 hashtags
    
    def _classify_segment_type(self, text: str) -> str:
        """Classify the type of segment"""
        text_lower = text.lower()
        
        if '?' in text:
            return 'question'
        elif any(word in text_lower for word in ['story', 'happened', 'time']):
            return 'story'
        elif any(word in text_lower for word in ['secret', 'reveal', 'truth']):
            return 'reveal'
        elif any(word in text_lower for word in ['problem', 'issue', 'challenge']):
            return 'climax'
        else:
            return 'hook'