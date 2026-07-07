"""
Content strategy helpers for turning generated clips into publish-ready assets.

The clipping pipeline finds moments; this module packages those moments with
titles, captions, hashtags, audience angles, CTAs, and thumbnail prompts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class ContentStrategyBuilder:
    """Build lightweight growth and publishing recommendations for each clip."""

    PLATFORM_DEFAULTS = {
        "youtube_shorts": {
            "name": "YouTube Shorts",
            "caption_limit": 140,
            "core_tags": ["#Shorts", "#YouTubeShorts"],
            "posting_window": "6-9 PM local time, then test 12-2 PM for your niche",
            "cta": "Subscribe for the next part.",
        },
        "tiktok": {
            "name": "TikTok",
            "caption_limit": 110,
            "core_tags": ["#TikTok", "#ForYou"],
            "posting_window": "7-10 PM local time, with a second test around lunch",
            "cta": "Follow for more clips like this.",
        },
        "instagram": {
            "name": "Instagram Reels",
            "caption_limit": 125,
            "core_tags": ["#Reels", "#InstagramReels"],
            "posting_window": "6-8 PM local time, especially Tue-Thu",
            "cta": "Save this and share it with someone who needs it.",
        },
        "linkedin": {
            "name": "LinkedIn",
            "caption_limit": 220,
            "core_tags": ["#Leadership", "#Business"],
            "posting_window": "8-10 AM local time on Tue-Thu",
            "cta": "Comment with the lesson you would add.",
        },
        "general": {
            "name": "Short-form",
            "caption_limit": 140,
            "core_tags": ["#Shorts", "#Viral"],
            "posting_window": "6-9 PM local time, then keep testing by audience",
            "cta": "Follow for the next clip.",
        },
    }

    STYLE_ANGLES = {
        "funny": {
            "promise": "a quick laugh with a relatable payoff",
            "cta": "Tag the friend who would react like this.",
            "thumbnail": "THIS WENT SIDEWAYS",
        },
        "educational": {
            "promise": "one clear lesson viewers can use immediately",
            "cta": "Save this so you can use it later.",
            "thumbnail": "DO THIS FIRST",
        },
        "energetic": {
            "promise": "a high-energy moment with momentum from the first second",
            "cta": "Send this to someone who needs the push.",
            "thumbnail": "LOCK IN",
        },
        "emotional": {
            "promise": "a human moment that earns attention through honesty",
            "cta": "Share this with someone who needs to hear it.",
            "thumbnail": "THIS HIT HARD",
        },
        "professional": {
            "promise": "a practical insight with business value",
            "cta": "Comment with your take on this.",
            "thumbnail": "SMART MOVE",
        },
        "viral": {
            "promise": "a surprising claim, tension, or payoff that invites comments",
            "cta": "Comment if you agree or disagree.",
            "thumbnail": "WAIT FOR IT",
        },
        "engaging": {
            "promise": "a self-contained moment with a clear reason to keep watching",
            "cta": "Follow for more moments like this.",
            "thumbnail": "WATCH THIS",
        },
    }

    STOPWORDS = {
        "about", "after", "again", "because", "before", "being", "could",
        "every", "from", "have", "into", "just", "like", "make", "more",
        "most", "people", "should", "that", "their", "there", "these",
        "thing", "this", "those", "through", "what", "when", "where",
        "which", "while", "with", "would", "your",
    }

    def build_for_clip(self, clip: Dict[str, Any], preferences: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return a publish-ready strategy package for a single clip."""
        preferences = preferences or {}
        platform_key = preferences.get("platform", "general")
        platform = self.PLATFORM_DEFAULTS.get(platform_key, self.PLATFORM_DEFAULTS["general"])
        style = preferences.get("style") or clip.get("emotion") or "engaging"
        style_data = self.STYLE_ANGLES.get(style, self.STYLE_ANGLES["engaging"])

        keywords = self._clean_keywords(clip)
        title = self._make_title(clip, keywords, style)
        hook = self._clean_sentence(clip.get("hook") or title)
        niche = preferences.get("niche") or self._infer_niche(keywords)
        goal = preferences.get("goal", "engagement")

        caption = self._fit_caption(
            f"{hook} {style_data['cta']}",
            platform["caption_limit"],
        )
        hashtags = self._hashtags(platform["core_tags"], keywords, niche, platform_key)

        return {
            "platform": platform["name"],
            "goal": goal,
            "audience_angle": self._audience_angle(niche, style_data["promise"], goal),
            "primary_title": title,
            "title_variants": self._title_variants(title, hook, keywords),
            "short_caption": caption,
            "long_caption": self._long_caption(hook, style_data["promise"], goal),
            "hashtags": hashtags,
            "seo_tags": self._seo_tags(keywords, niche),
            "thumbnail_text": self._thumbnail_text(clip, style_data),
            "thumbnail_prompt": self._thumbnail_prompt(clip, style_data, niche),
            "call_to_action": style_data["cta"] if platform_key == "general" else platform["cta"],
            "posting_window": platform["posting_window"],
            "engagement_question": self._engagement_question(style, niche),
            "virality_reasons": self._virality_reasons(clip, style_data),
            "repurpose_notes": self._repurpose_notes(platform_key),
        }

    def build_for_clips(self, clips: List[Dict[str, Any]], preferences: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Attach content packages to every clip and return the same clip list."""
        for clip in clips:
            clip["content_package"] = self.build_for_clip(clip, preferences)
        return clips

    def _make_title(self, clip: Dict[str, Any], keywords: List[str], style: str) -> str:
        base = self._clean_title(clip.get("title") or "")
        if base and not base.lower().startswith("engaging moment"):
            return base[:70]

        if style == "educational":
            lead = "The lesson nobody explains"
        elif style == "professional":
            lead = "The strategy that changes everything"
        elif style == "funny":
            lead = "This moment went completely wrong"
        elif style == "emotional":
            lead = "The moment that changed the room"
        else:
            lead = "The moment everyone will replay"

        if keywords:
            return f"{lead}: {keywords[0].title()}"[:70]
        return lead

    def _title_variants(self, title: str, hook: str, keywords: List[str]) -> List[str]:
        key = keywords[0].title() if keywords else "This"
        hook_short = self._fit_caption(hook, 58)
        variants = [
            title,
            f"Nobody expected this about {key}"[:70],
            f"{hook_short}"[:70],
        ]
        return list(dict.fromkeys(v for v in variants if v))

    def _hashtags(self, core_tags: List[str], keywords: List[str], niche: str, platform_key: str) -> List[str]:
        tags = list(core_tags)
        for word in keywords[:5]:
            tag = "#" + re.sub(r"[^A-Za-z0-9]", "", word.title())
            if len(tag) > 1:
                tags.append(tag)
        if niche and niche != "your niche":
            tags.append("#" + re.sub(r"[^A-Za-z0-9]", "", niche.title()))
        if platform_key == "youtube_shorts":
            tags.append("#CreatorTips")
        return list(dict.fromkeys(tags))[:10]

    def _seo_tags(self, keywords: List[str], niche: str) -> List[str]:
        tags = keywords[:8]
        tags.extend(["shorts", "viral clip", "creator growth", niche])
        return [tag for tag in list(dict.fromkeys(tags)) if tag][:12]

    def _thumbnail_text(self, clip: Dict[str, Any], style_data: Dict[str, str]) -> str:
        title = self._clean_title(clip.get("title") or "")
        if title and len(title.split()) <= 4:
            return title.upper()
        return style_data["thumbnail"]

    def _thumbnail_prompt(self, clip: Dict[str, Any], style_data: Dict[str, str], niche: str) -> str:
        text = self._thumbnail_text(clip, style_data)
        return (
            "Create a 9:16 short-form thumbnail frame with bold readable text "
            f"'{text}', high-contrast subject focus, expressive face or clear action, "
            f"and visual cues for {niche}. Avoid clutter."
        )

    def _audience_angle(self, niche: str, promise: str, goal: str) -> str:
        if goal == "subscribers":
            return f"Attract {niche} viewers by promising {promise}, then ask them to follow the series."
        if goal == "sales":
            return f"Warm up {niche} buyers with {promise}, then bridge to the offer in comments or description."
        return f"Pull in {niche} viewers with {promise} and invite a simple comment."

    def _long_caption(self, hook: str, promise: str, goal: str) -> str:
        return (
            f"{hook}\n\n"
            f"Why it works: this clip gives viewers {promise} without needing the full video.\n\n"
            f"Goal: optimize for {goal}. Watch the first 3 seconds, retention, comments, and saves."
        )

    def _engagement_question(self, style: str, niche: str) -> str:
        if style == "funny":
            return "Be honest, would you have laughed in this situation?"
        if style == "educational":
            return f"What is one {niche} tip you wish you learned earlier?"
        if style == "professional":
            return "What would you do differently here?"
        if style == "emotional":
            return "Have you ever had a moment like this?"
        return "Do you agree with this take?"

    def _virality_reasons(self, clip: Dict[str, Any], style_data: Dict[str, str]) -> List[str]:
        score = float(clip.get("engagement_score") or 0)
        reasons = ["Self-contained hook for cold viewers", style_data["promise"].capitalize()]
        if score >= 8:
            reasons.append("High selection score from transcript analysis")
        if clip.get("hashtags"):
            reasons.append("Already has topic tags for distribution")
        return reasons[:4]

    def _repurpose_notes(self, platform_key: str) -> List[str]:
        if platform_key == "linkedin":
            return ["Post the clip with a text lesson", "Turn the caption into a carousel opener"]
        if platform_key == "instagram":
            return ["Use as a Reel", "Share the best frame to Stories with a poll"]
        if platform_key == "tiktok":
            return ["Post a direct cut first", "Reply to the strongest comment with part two"]
        return ["Post as Shorts/Reels/TikTok", "Use the hook as a community post question"]

    def _clean_keywords(self, clip: Dict[str, Any]) -> List[str]:
        raw = []
        raw.extend(clip.get("keywords") or [])
        raw.extend((clip.get("title") or "").split())
        raw.extend((clip.get("hook") or "").split())

        cleaned = []
        for item in raw:
            word = re.sub(r"[^A-Za-z0-9 ]", "", str(item)).strip().lower()
            if not word or len(word) < 4 or word in self.STOPWORDS:
                continue
            if word not in cleaned:
                cleaned.append(word)
        return cleaned[:10]

    def _infer_niche(self, keywords: List[str]) -> str:
        if not keywords:
            return "your niche"
        if any(k in keywords for k in ["business", "marketing", "sales", "strategy"]):
            return "business creators"
        if any(k in keywords for k in ["podcast", "interview", "story"]):
            return "podcast audiences"
        if any(k in keywords for k in ["learn", "lesson", "tutorial", "teach"]):
            return "learners"
        return f"{keywords[0]} fans"

    def _clean_title(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value)).strip(" -_")

    def _clean_sentence(self, value: str) -> str:
        value = re.sub(r"\s+", " ", str(value)).strip()
        return value.rstrip(".") + "." if value and value[-1] not in ".!?" else value

    def _fit_caption(self, value: str, limit: int) -> str:
        value = re.sub(r"\s+", " ", str(value)).strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)].rstrip() + "..."
