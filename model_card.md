# 🎧 Model Card: Context-Aware AI DJ Music Recommender

## 1. Model Name  

**VibeRank v2.0 — Context-Aware AI DJ**

---

## 2. Intended Use  

This is a classroom simulation built to show how content-based filtering works under the hood, extended with a full RAG (Retrieval-Augmented Generation) pipeline. It recommends a ranked list of songs from a small dataset based on a user's genre, mood, energy preferences, and current activity context (e.g. studying, commuting, working out). For each recommendation, it retrieves artist facts from a local knowledge base, verifies them with a guardrail, and uses an LLM to generate a personalised DJ-style introduction. It is not meant for production use — it is a learning system designed to demonstrate how RAG, guardrails, confidence scoring, and reliability testing work together.

---

## 3. How the Model Works  

The system works in two modes:

**Mode 1 — Content-based scoring:** Each song is scored against the user's profile using a weighted formula: genre match (+2.0), mood match (+1.0), energy proximity (2.0 × closeness), and acoustic preference (+0.5). Songs are ranked highest to lowest and the top-k are returned with score explanations.

**Mode 2 — RAG + LLM DJ:** For each top-k song, the system looks up an artist bio from a local `knowledge_base/` folder, verifies it with a guardrail check (artist name must appear in the bio), computes a confidence score (0.0–1.0), and calls an LLM to generate a contextual DJ introduction that connects the song's qualities to the user's current activity.

---

## 4. Data  

The dataset has 20 songs stored in a CSV file with 10 audio features each: genre, mood, energy, tempo, valence, danceability, and acousticness. The dataset was expanded from the original 15 tracks to include melancholic rock tracks (Radiohead, Soundgarden, Johnny Cash, Green Day) and more diverse genres (reggae, soul/r&b, dark ambient, city pop).

The knowledge base contains 6 artist bio `.txt` files written manually. Artists without a bio file get 0% confidence and metadata-only DJ scripts.

The data is still Western/mainstream-heavy and does not capture lyrics, subgenres, cultural context, or listening history.

---

## 5. Strengths  

The system works well when the user's preferences align with the dataset. The Melancholic Rock profile consistently pulls accurate sad rock songs at the top. The RAG pipeline meaningfully improves DJ script quality — scripts for artists with bios cite real facts (e.g. Green Day's American Idiot, Chris Cornell's vocals) while scripts without bios fall back cleanly to metadata without hallucinating.

The guardrail prevents wrong bios from reaching the LLM, and the confidence score gives a transparent measure of how much the system knows about each recommendation.

---

## 6. Limitations and Bias

**Label bias:** Genre is weighted the most (+2.0), so the system will sometimes prioritise genre match over mood match. A user who wants "sad" music but specifies "pop" will get upbeat pop songs ranked higher than sad songs from other genres.

**Diversity problem:** The dataset is small (20 songs) so the same songs appear repeatedly across profiles.

**Knowledge base gaps:** Only 6 of 20 artists have bio files. The other 14 generate 0% confidence scores and generic scripts.

**No cultural or emotional context:** The system treats "Hurt" by Johnny Cash as just numbers. Small numerical differences can outweigh actual human meaning.

**Western bias:** The dataset skews toward Western popular music and doesn't represent global music traditions.

---

## 7. Evaluation  

**Automated testing:** 23/23 pytest tests pass across 5 categories: scoring logic, bio retrieval, guardrail verification, confidence scoring, and CSV loading. The 3 initially failing tests revealed a real bug — the `Recommender` class was passing a `UserProfile` object instead of a dict to `score_song()`. Fixing this brought the suite to 100%.

**Confidence scoring:** Averaged 0% without knowledge base files, 50–80% after adding artist bios. Boulevard of Broken Dreams scored 80% confidence in the Working Out profile because the Green Day bio contained energy-relevant keywords and the song's tempo matched the activity range.

**Guardrail results:** 4 passes and 8 fallbacks across a full run of 12 recommendations. All 8 fallbacks generated coherent scripts from metadata alone — no hallucinated facts.

**Adversarial profile testing:** The "High-Energy Weeper" profile (sad mood + 0.95 energy) showed the system correctly surfacing the genre/energy tension — the LLM even acknowledged it: "I know you're feeling down today, but this one's going to get you moving."

---

## 8. Reflection and Ethics

**What are the limitations or biases in your system?**
The biggest bias is label bias from the genre weight. Because genre is worth +2.0 and mood is only +1.0, a genre match always outweighs a mood mismatch. This means a user who explicitly wants "sad" music but lists "pop" as their genre will get happy pop songs ranked above sad rock songs. The small dataset also creates a diversity problem — the same 5–6 songs dominate every profile. The knowledge base coverage gap (6 of 20 artists) means most recommendations lack contextual grounding.

**Could your AI be misused, and how would you prevent that?**
The system is low-risk since it only recommends songs. However, a similar RAG architecture applied to medical, legal, or financial advice could cause real harm if bios were fabricated or guardrails were bypassed. The guardrail pattern used here — verify retrieved content before passing it to the LLM — is exactly the right prevention mechanism. Logging every recommendation with confidence scores also creates an audit trail.

**What surprised you while testing your AI's reliability?**
The most surprising finding was how the confidence score exposed the system's own knowledge gaps in real time. Seeing 0% confidence for 14 out of 20 artists made it immediately obvious that the system was generating DJ scripts from thin air for most recommendations. This directly motivated adding more knowledge base files. The other surprise was how the adversarial profile produced the most interesting LLM outputs — the model detected the contradiction between mood and energy and surfaced it in the script, which was not explicitly prompted.

**Describe your collaboration with AI during this project. Identify one instance when the AI gave a helpful suggestion and one instance where its suggestion was flawed or incorrect.**

*Helpful suggestion:* When designing the RAG pipeline, the AI suggested adding a `guardrail_check()` function that verifies the artist's name appears in the bio before passing it to the LLM. This was genuinely useful — without it, a misnamed file (e.g. `radiohead.txt` containing a Coldplay bio) would silently produce confident-sounding but wrong DJ scripts. The guardrail caught this class of error entirely.

*Flawed suggestion:* The AI repeatedly suggested using `from src.recommender import` in `main.py` even though `main.py` lives inside `src/` and is run from there directly. This caused a `ModuleNotFoundError: No module named 'src'` that took significant time to debug. The correct import was simply `from recommender import` since Python's path already included the `src/` directory when running from inside it. The AI's suggestion was technically correct for one running style but wrong for how the project was actually structured.

---

## 9. Future Work  

- Adding a "serendipity" factor so users don't get stuck in the same vibe loop
- Expanding the knowledge base to cover all 20 artists
- Using acoustic features like `acousticness` to better separate instrumental vs. electronic tracks
- Improving explanations so the system can say "this is similar to your favourite artist" instead of raw score breakdowns
- Adding an end-to-end integration test that mocks the LLM and asserts the DJ script contains expected artist facts

---

## 10. Personal Reflection  

This project made it clear that recommendation systems aren't actually "understanding" music — they're doing fast math on patterns. What surprised me most is how easy it is to accidentally introduce bias by tweaking a weight slightly. Changing 1.5 to 2.0 completely shifted what the system considered "good."

The RAG extension changed how I think about AI reliability. Retrieved context is only as trustworthy as the system that verifies it. The guardrail pattern — retrieve, verify, then generate — is the right order of operations for any system that needs to be trusted. Building it myself made that principle concrete in a way that reading about it never would have.