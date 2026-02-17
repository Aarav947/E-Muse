# E-Muse
Real-time music recommendation system that detects your facial emotion via webcam and suggests songs matched to your emotional state using Spotify audio features.

What It Does
Captures webcam feed → detects face → classifies emotion → maps to music mood profile → filters and ranks songs by audio feature similarity.

ML Pipeline
Step 1 — Face Detection
MediaPipe detects face and extracts ROI from each video frame.
Step 2 — Emotion Classification
Face preprocessed (grayscale, 48×48, normalized) and passed through a CNN producing a probability distribution across 7 emotions.

Step 3 — Emotion → Music Mapping
Detected emotion mapped to target Spotify audio features:
Happy → high valence, high energy
Sad → low valence, low energy
Angry → high energy, moderate valence

Step 4 — Recommendation Filtering
Songs filtered by proximity to target mood profile across valence, energy, and danceability. Top matches ranked by feature similarity.

Key Features
Real-time emotion detection from live webcam feed
7-emotion classification using CNN
Spotify audio feature-based filtering (valence, energy, danceability)
Similarity-based song ranking


Tech Stack
Python · MediaPipe · CNN · Streamlit · Spotify Dataset · OpenCV · Pandas

<img width="2443" height="1233" alt="image" src="https://github.com/user-attachments/assets/189376f8-4ea1-4fe8-9a90-a43f680abf49" />

<img width="1389" height="1252" alt="image" src="https://github.com/user-attachments/assets/331f635d-d1a2-47b5-bcc4-32c8189f0b5f" />


