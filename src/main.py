"""
Command line runner for the Context-Aware AI DJ Music Recommender.

Runs two modes:
  1. Quick score test  — uses recommend_songs() for fast offline testing
  2. Full DJ mode      — uses recommend_with_dj() which calls Claude and
                         reads from knowledge_base/ for RAG-generated scripts

Toggle between modes by commenting/uncommenting the relevant block below.
"""

from recommender import load_songs, recommend_songs, recommend_with_dj

def main() -> None:
    songs = load_songs("../data/songs.csv")

    # -----------------------------------------------------------------------
    # Profiles — updated to use favorite_genres (list) and current_context
    # -----------------------------------------------------------------------
    profiles = [
        {
            "name": "Default Pop",
            "favorite_genres": ["pop"],
            "favorite_mood": "happy",
            "target_energy": 0.8,
            "likes_acoustic": False,
            "current_context": "Commuting",
        },
        {
            "name": "Melancholic Rock",
            "favorite_genres": ["rock"],
            "favorite_mood": "sad",
            "target_energy": 0.55,
            "likes_acoustic": False,
            "current_context": "Relaxing",
        },
        {
            "name": "Chill Lofi",
            "favorite_genres": ["lofi"],
            "favorite_mood": "chill",
            "target_energy": 0.2,
            "likes_acoustic": True,
            "current_context": "Studying",
        },
        {
            "name": "Adversarial: The High-Energy Weeper",
            "favorite_genres": ["pop"],
            "favorite_mood": "sad",
            "target_energy": 0.95,
            "likes_acoustic": False,
            "current_context": "Working out",
        },
    ]

    # -----------------------------------------------------------------------
    # MODE 1 — Quick offline scoring test (no API calls)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("MODE 1 — Score-based recommendations (no AI)")
    print("=" * 60)

    for profile in profiles:
        print(f"\n--- {profile['name']} ---")
        recommendations = recommend_songs(profile, songs, k=5)
        for song, score, reasons in recommendations:
            print(f"  {song['title']} by {song['artist']}  |  Score: {score:.2f}")
            print(f"  Because: {reasons}\n")

    # -----------------------------------------------------------------------
    # MODE 2 — Full AI DJ mode (calls Claude + reads knowledge_base/)
    # Uncomment this block to run the RAG pipeline.
    # Requires google api to be set in your environment.
    # -----------------------------------------------------------------------
    
    print("=" * 60)
    print("MODE 2 — Context-Aware AI DJ (RAG + Claude)")
    print("=" * 60)

    for profile in profiles:
        print(f"\n--- DJ Set for: {profile['name']} ---")
        dj_results = recommend_with_dj(
            user_prefs=profile,
            songs=songs,
            knowledge_base_dir="../knowledge_base",
            k=3,
        )
        for song, score, dj_script, confidence in dj_results:
            print(f"  {song['title']} by {song['artist']}  |  Score: {score:.2f}  |  Confidence: {confidence:.0%}")
            print(f"  🎙️  {dj_script}\n")
    


if __name__ == "__main__":
    main()