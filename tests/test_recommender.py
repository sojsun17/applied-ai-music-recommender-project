"""
Tests for the Music Recommender.
Run with: pytest test_recommender.py -v

Covers:
  - Song dataclass construction
  - UserProfile dataclass construction
  - Recommender.recommend() sorting
  - Recommender.explain_recommendation() output
  - score_song() — genre, mood, energy, acoustic bonus
  - load_artist_bio() — hit, miss, slug conversion
  - guardrail_check() — pass, fail, empty
  - compute_confidence_score() — zero without bio, scaling with context
"""

import os
import tempfile
import pytest

# Import path matches your project layout (no src/ prefix)
from recommender  import (
    Song,
    UserProfile,
    Recommender,
    score_song,
    load_artist_bio,
    guardrail_check,
    compute_confidence_score,
    load_songs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_song(**overrides) -> Song:
    defaults = dict(
        id=1, title="Test Track", artist="Test Artist",
        genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
        valence=0.8, danceability=0.75, acousticness=0.2,
    )
    defaults.update(overrides)
    return Song(**defaults)


def make_user(**overrides) -> UserProfile:
    defaults = dict(
        favorite_genres=["pop"],
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
        current_context="Studying",
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
            valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
            valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = make_user()
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Pop/happy/high-energy song should rank above lofi/chill
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_respects_k():
    user = make_user()
    rec = make_small_recommender()
    assert len(rec.recommend(user, k=1)) == 1


def test_explain_recommendation_returns_non_empty_string():
    user = make_user()
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# score_song
# ---------------------------------------------------------------------------

def test_score_genre_match_adds_two_points():
    user = make_user(favorite_genres=["pop"])
    song = make_song(genre="pop", mood="sad", energy=0.5).__dict__
    score, reasons = score_song(user.__dict__, song)
    assert score >= 2.0
    assert any("Genre match" in r for r in reasons)


def test_score_genre_mismatch_no_bonus():
    user = make_user(favorite_genres=["rock"])
    song = make_song(genre="pop", mood="sad", energy=0.5).__dict__
    _, reasons = score_song(user.__dict__, song)
    assert not any("Genre match" in r for r in reasons)


def test_score_multi_genre_match():
    user = make_user(favorite_genres=["rock", "pop"])
    song = make_song(genre="pop").__dict__
    _, reasons = score_song(user.__dict__, song)
    assert any("Genre match" in r for r in reasons)


def test_score_mood_match_adds_one_point():
    user = make_user(favorite_mood="chill")
    song_match = make_song(genre="jazz", mood="chill", energy=0.4).__dict__
    song_no    = make_song(genre="jazz", mood="intense", energy=0.4).__dict__
    score_m, _ = score_song(user.__dict__, song_match)
    score_n, _ = score_song(user.__dict__, song_no)
    assert score_m - score_n == pytest.approx(1.0)


def test_score_perfect_energy_gives_two():
    user = make_user(target_energy=0.7)
    song = make_song(genre="jazz", mood="sad", energy=0.7).__dict__
    _, reasons = score_song(user.__dict__, song)
    energy_reason = next(r for r in reasons if "Energy" in r)
    assert "+2.00" in energy_reason


def test_score_acoustic_bonus_applied():
    user = make_user(likes_acoustic=True)
    song_ac  = make_song(acousticness=0.9).__dict__
    song_not = make_song(acousticness=0.1).__dict__
    score_ac, _ = score_song(user.__dict__, song_ac)
    score_no, _ = score_song(user.__dict__, song_not)
    assert score_ac > score_no


def test_score_no_acoustic_bonus_when_preference_false():
    user = make_user(likes_acoustic=False)
    song = make_song(acousticness=0.95).__dict__
    _, reasons = score_song(user.__dict__, song)
    assert not any("Acoustic" in r for r in reasons)


# ---------------------------------------------------------------------------
# load_artist_bio
# ---------------------------------------------------------------------------

def test_load_bio_returns_content_when_file_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "radiohead.txt")
        with open(path, "w") as f:
            f.write("Radiohead are an English rock band.")
        result = load_artist_bio(tmpdir, "Radiohead")
        assert result is not None
        assert "Radiohead" in result


def test_load_bio_returns_none_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert load_artist_bio(tmpdir, "Nonexistent Artist") is None


def test_load_bio_slug_converts_spaces_and_caps():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "johnny_cash.txt")
        with open(path, "w") as f:
            f.write("Johnny Cash was a country legend.")
        result = load_artist_bio(tmpdir, "Johnny Cash")
        assert result is not None


def test_load_bio_slug_handles_special_characters():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "soul_r_b_artist.txt")
        with open(path, "w") as f:
            f.write("A soul and R&B artist.")
        result = load_artist_bio(tmpdir, "Soul R&B Artist")
        assert result is not None


# ---------------------------------------------------------------------------
# guardrail_check
# ---------------------------------------------------------------------------

def test_guardrail_passes_when_name_in_bio():
    assert guardrail_check("Radiohead", "Radiohead formed in Abingdon in 1985.") is True


def test_guardrail_fails_when_name_absent():
    assert guardrail_check("Radiohead", "This band formed in Seattle in 1990.") is False


def test_guardrail_fails_on_empty_bio():
    assert guardrail_check("Radiohead", "") is False


def test_guardrail_case_insensitive():
    assert guardrail_check("Johnny Cash", "johnny cash was born in Arkansas.") is True


# ---------------------------------------------------------------------------
# compute_confidence_score
# ---------------------------------------------------------------------------

def test_confidence_zero_without_bio():
    song = make_song(energy=0.4).__dict__
    assert compute_confidence_score(None, "Studying", song) == 0.0


def test_confidence_at_least_half_with_any_bio():
    song = make_song(energy=0.4).__dict__
    assert compute_confidence_score("Some bio text.", "Studying", song) >= 0.5


def test_confidence_increases_with_relevant_keywords():
    song = make_song(energy=0.45).__dict__
    generic = compute_confidence_score("A great artist.", "Studying", song)
    relevant = compute_confidence_score(
        "Known for calm, instrumental, ambient focus music.", "Studying", song
    )
    assert relevant > generic


def test_confidence_max_one():
    bio = "focus calm steady ambient instrumental"
    song = make_song(energy=0.45).__dict__
    assert compute_confidence_score(bio, "Studying", song) <= 1.0


# ---------------------------------------------------------------------------
# load_songs
# ---------------------------------------------------------------------------

def test_load_songs_returns_correct_count():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n")
        f.write("1,Alpha,ArtistA,pop,happy,0.8,120,0.9,0.75,0.2\n")
        f.write("2,Beta,ArtistB,rock,sad,0.6,100,0.4,0.5,0.3\n")
        fname = f.name
    try:
        songs = load_songs(fname)
        assert len(songs) == 2
        assert songs[0]["title"] == "Alpha"
        assert isinstance(songs[0]["energy"], float)
    finally:
        os.unlink(fname)