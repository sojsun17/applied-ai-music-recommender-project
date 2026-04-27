from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by test_recommender.py

    favorite_genres accepts a list so users can match across multiple genres
    (e.g. ["rock", "indie pop"]).  The old single-string field is gone —
    update any callers to pass a list, even if it contains just one item.
    """
    favorite_genres: List[str]       # was favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    current_context: str = "Listening"  # e.g. "Studying", "Working out"


# ---------------------------------------------------------------------------
# Recommender class  (used by test_recommender.py)
# ---------------------------------------------------------------------------

class Recommender:
    """
    OOP wrapper around the recommendation logic.
    Required by test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Songs sorted by score, highest first."""
        scored = []
        for song in self.songs:
            score, _ = score_song(user.__dict__, song.__dict__)
            scored.append((song, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation for why this song was recommended."""
        _, reasons = score_song(user.__dict__, song.__dict__)
        return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Core functions  (used by main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file and return them as a list of dicts.
    Required by main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Compute a recommendation score for one song against user preferences.

    Scoring recipe:
      - Genre match:   +2.0 if the song's genre is in the user's favorite_genres list
      - Mood match:    +1.0 if moods match exactly
      - Energy score:  2.0 * (1.0 - abs(target_energy - song_energy))
      - Acoustic bonus:+0.5 if likes_acoustic is True and song acousticness > 0.6

    Args:
        user_prefs: Dict with keys "favorite_genres" (list[str]), "favorite_mood",
                    "target_energy", and optionally "likes_acoustic".
        song:       Dict with keys "genre", "mood", "energy", "acousticness".

    Returns:
        (total_score, reasons) — score as float, reasons as list of strings.
    """
    total_score = 0.0
    reasons: List[str] = []

    # Support both old single-string key and new list key so nothing breaks
    favorite_genres = user_prefs.get("favorite_genres") or []
    if isinstance(favorite_genres, str):
        favorite_genres = [favorite_genres]

    if song["genre"] in favorite_genres:
        total_score += 2.0
        reasons.append("Genre match (+2.0)")

    if song["mood"] == user_prefs["favorite_mood"]:
        total_score += 1.0
        reasons.append("Mood match (+1.0)")

    energy_score = 2.0 * (1.0 - abs(user_prefs["target_energy"] - float(song["energy"])))
    total_score += energy_score
    reasons.append(f"Energy match x2.0 (+{energy_score:.2f})")

    if user_prefs.get("likes_acoustic") and float(song.get("acousticness", 0)) > 0.6:
        total_score += 0.5
        reasons.append("Acoustic preference match (+0.5)")

    return total_score, reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Score and rank songs against user preferences, return the top-k.

    Returns a list of (song_dict, score, explanation_string) tuples,
    ordered highest score first.
    """
    scored_songs: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored_songs.append((song, score, explanation))

    ranked = sorted(scored_songs, key=lambda item: item[1], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def load_artist_bio(knowledge_base_dir: str, artist_name: str) -> Optional[str]:
    """
    Look up an artist bio from the knowledge_base folder.
    File naming: artist name lowercased, non-alphanumeric runs → underscore.
    e.g. "Johnny Cash" → knowledge_base/johnny_cash.txt
    """
    slug = re.sub(r"[^a-z0-9]+", "_", artist_name.lower()).strip("_")
    path = Path(knowledge_base_dir) / f"{slug}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logger.warning(f"No bio found for '{artist_name}' at {path}")
    return None


def guardrail_check(artist_name: str, bio_text: str) -> bool:
    """
    Verify the bio actually belongs to the expected artist by checking
    that at least one meaningful name part appears in the bio text.
    Returns False if bio_text is empty or None.
    """
    if not bio_text:
        return False
    name_parts = artist_name.lower().split()
    bio_lower = bio_text.lower()
    return any(part in bio_lower for part in name_parts if len(part) > 2)


def compute_confidence_score(
    bio_text: Optional[str],
    context: str,
    song: Dict,
) -> float:
    """
    Heuristic 0.0–1.0 confidence score for a recommendation:
      +0.5  bio was found and passed guardrail
      +0.3  bio contains keywords relevant to the user's activity
      +0.2  song energy falls in the expected range for that activity
    """
    if not bio_text:
        return 0.0

    confidence = 0.5

    context_keywords = {
        "studying":    ["focus", "steady", "ambient", "instrumental", "calm"],
        "working out": ["energy", "intense", "beat", "rhythm", "powerful"],
        "commuting":   ["city", "journey", "movement", "urban", "flow"],
        "sleeping":    ["soft", "gentle", "calm", "slow", "quiet"],
        "relaxing":    ["chill", "mellow", "smooth", "gentle"],
    }
    context_energy_ranges = {
        "studying":    (0.3, 0.6),
        "working out": (0.7, 1.0),
        "commuting":   (0.4, 0.8),
        "sleeping":    (0.0, 0.3),
        "relaxing":    (0.2, 0.5),
    }

    keywords = context_keywords.get(context.lower(), [])
    if any(kw in bio_text.lower() for kw in keywords):
        confidence += 0.3

    low, high = context_energy_ranges.get(context.lower(), (0.0, 1.0))
    if low <= float(song.get("energy", 0.5)) <= high:
        confidence += 0.2

    return min(confidence, 1.0)

def generate_dj_script(
    song: Dict,
    user_prefs: Dict,
    bio_text: Optional[str],
) -> str:
    """
    Call an LLM to produce a contextual 2-3 sentence DJ introduction for a song.
    Falls back to a plain score-based explanation if the API call fails.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    bio_section = f"\nArtist bio:\n{bio_text}" if bio_text else "\nNo artist bio available."
    context = user_prefs.get("current_context", "Listening")
    genres = user_prefs.get("favorite_genres", [])

    prompt = f"""You are a witty, knowledgeable AI DJ introducing a song to a listener.

User context:
- Current activity: {context}
- Mood preference: {user_prefs.get('favorite_mood', 'any')}
- Favourite genres: {', '.join(genres) if genres else 'varied'}

Song being introduced:
- Title: "{song['title']}" by {song['artist']}
- Genre: {song['genre']}
- Mood: {song['mood']}
- Energy: {float(song['energy']):.2f} / 1.0
- Tempo: {song['tempo_bpm']} BPM
- Acousticness: {float(song.get('acousticness', 0)):.2f}
{bio_section}

Write a 2-3 sentence DJ introduction that:
1. Connects the song's specific qualities (energy, tempo, mood) to the user's current activity ("{context}")
2. Cites at least one concrete fact from the artist bio if available
3. Sounds like a real radio DJ — enthusiastic but not over the top

Respond with only the DJ script, no preamble."""

    try:
        response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"DJ script generation failed for '{song['title']}': {e}")
        _, reasons = score_song(user_prefs, song)
        return "Recommended because: " + "; ".join(reasons)


# ---------------------------------------------------------------------------
# Full RAG recommend pipeline  (called by main.py)
# ---------------------------------------------------------------------------

def recommend_with_dj(
    user_prefs: Dict,
    songs: List[Dict],
    knowledge_base_dir: str,
    k: int = 3,
) -> List[Tuple[Dict, float, str, float]]:
    """
    Full Context-Aware AI DJ pipeline.

    1. Score every song with score_song()
    2. Take top-k
    3. For each: load bio → guardrail check → generate DJ script → confidence score
    4. Log everything to system.log

    Returns a list of (song_dict, score, dj_script, confidence_score) tuples.
    """
    top_k = recommend_songs(user_prefs, songs, k=k)
    results = []

    for song, score, _ in top_k:
        bio = load_artist_bio(knowledge_base_dir, song["artist"])

        if bio and not guardrail_check(song["artist"], bio):
            logger.error(
                f"GUARDRAIL FAILED — bio for '{song['artist']}' may be mismatched. "
                "Falling back to no-bio generation."
            )
            bio = None

        dj_script = generate_dj_script(song, user_prefs, bio)
        confidence = compute_confidence_score(
            bio, user_prefs.get("current_context", "Listening"), song
        )

        logger.info(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "song": f"{song['title']} — {song['artist']}",
            "score": round(score, 3),
            "context": user_prefs.get("current_context", "Listening"),
            "confidence": round(confidence, 3),
            "guardrail_passed": bio is not None,
            "bio_available": bio is not None,
        }))

        results.append((song, score, dj_script, confidence))

    return results