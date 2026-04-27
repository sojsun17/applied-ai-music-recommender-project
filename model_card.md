# 🎧 Context-Aware AI DJ

A RAG-powered music recommender that acts as a real DJ — it doesn't just score songs, it *explains* why each track fits your current vibe by citing facts from a local knowledge base and generating contextual introductions via Claude.

---

## Project Structure

```
dj_recommender/
├── app.py                  # Streamlit web UI
├── rag_recommender.py      # Core RAG + scoring engine
├── test_rag_recommender.py # Pytest test suite
├── requirements.txt
├── system.log              # Auto-generated recommendation log
├── data/
│   └── songs.csv           # Song library (20 tracks)
└── knowledge_base/         # Artist bio .txt files for RAG
    ├── radiohead.txt
    ├── johnny_cash.txt
    ├── neon_echo.txt
    ├── loroom.txt
    ├── green_day.txt
    └── soundgarden.txt
```

---

## How It Works

### 1. Scoring (Content-Based Filtering)
Each song is scored against the user's `UserProfile`:
- **Genre match**: +2.0 pts (supports multi-genre preferences)
- **Mood match**: +1.0 pt
- **Energy proximity**: `2.0 × (1 - |target - song_energy|)` — max +2.0
- **Acoustic bonus**: +0.5 if user prefers acoustic and song acousticness > 0.6

### 2. RAG — Knowledge Base Retrieval
For each top-k song, the system looks up `knowledge_base/{artist_snake_case}.txt`.  
If found, the bio is passed to Claude alongside the user's context to generate a personalised DJ introduction.

### 3. Guardrail Check
Before using a bio, the system checks that the artist's name appears in the bio text. If it doesn't (mismatched file), the bio is discarded and generation falls back to song metadata only. This prevents "hallucinating" facts about the wrong artist.

### 4. DJ Script Generation (LLM)
Claude generates a 2–3 sentence intro that:
- Connects song qualities (energy, tempo, mood) to the user's current activity
- Cites at least one fact from the artist bio when available
- Reads like a real radio DJ

### 5. Confidence Score
Each recommendation gets a `0.0–1.0` confidence score:
- Bio available: +0.5
- Bio contains context-relevant keywords: +0.3
- Song energy fits the activity: +0.2

### 6. Logging
Every recommendation is logged to `system.log` as a JSON line with timestamp, song, score, context, confidence, and guardrail result.

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

### Run the Web App
```bash
streamlit run app.py
```

### Run Tests
```bash
pytest test_rag_recommender.py -v
```

---

## Adding Artists to the Knowledge Base

Create a file in `knowledge_base/` named `{artist_name_snake_case}.txt`.

Examples:
- `the_weeknd.txt` for "The Weeknd"
- `taylor_swift.txt` for "Taylor Swift"
- `orbit_bloom.txt` for "Orbit Bloom"

The guardrail will auto-verify the file content contains the artist's name.

---

## Extending the Song Library

Add rows to `data/songs.csv` following the schema:
```
id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness
```
- `energy`, `valence`, `danceability`, `acousticness`: float 0.0–1.0
- `tempo_bpm`: integer
- `mood`: free text (happy, sad, chill, intense, focused, moody, etc.)