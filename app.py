import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emotion_detector import EmotionDetector
from spotify_recommender import SpotifyRecommender
from recommendation_engine import RecommendationEngine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="E-Muse: Emotion-Driven Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1DB954;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
        margin: 10px 5px;
    }
    .emotion-happy { background-color: #FFD700; color: #000; }
    .emotion-sad { background-color: #4169E1; color: #fff; }
    .emotion-angry { background-color: #FF4500; color: #fff; }
    .emotion-neutral { background-color: #808080; color: #fff; }
    .emotion-surprised { background-color: #FF69B4; color: #fff; }
    .emotion-fearful { background-color: #9932CC; color: #fff; }
    .emotion-disgusted { background-color: #228B22; color: #fff; }
    .recommendation-card {
        background-color: var(--background-color);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1DB954;
        border: 1px solid rgba(140, 140, 140, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("## ⚙️ Configuration")

# Webcam Settings
st.sidebar.markdown("### 📷 Webcam Settings")
confidence_threshold = st.sidebar.slider(
    "Emotion Detection Confidence Threshold",
    min_value=0.3,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="The detection will stop once an emotion is detected above this confidence level."
)

# Recommendation Settings
st.sidebar.markdown("### 🎼 Recommendation Settings")
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

energy_weight = st.sidebar.slider(
    "Energy Weight (0=Low, 1=High)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Controls how 'energetic' the recommendations should be"
)

valence_weight = st.sidebar.slider(
    "Valence Weight (0=Sad, 1=Happy)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Controls how 'positive' the recommendations should be"
)

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
st.markdown('<div class="main-header">🎵 E-Muse: Emotion-Driven Music Recommendations</div>', unsafe_allow_html=True)
st.markdown("Detect your emotion in real-time and get personalized music recommendations based on your mood!")

# Check if dataset is present
music_data_path = os.path.join(os.path.dirname(__file__), "music_data.csv")
if not os.path.exists(music_data_path):
    st.error(f"❌ Dataset not found! Please ensure 'music_data.csv' exists in the same directory as app.py.")
    st.stop()

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "emotion_detector" not in st.session_state:
    st.session_state.emotion_detector = None

emotion_model_path = os.getenv("EMOTION_MODEL_PATH", "emotion_model.h5")

if "spotify_recommender" not in st.session_state:
    st.session_state.spotify_recommender = None

if "recommendation_engine" not in st.session_state:
    st.session_state.recommendation_engine = None

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = None

if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = None

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

# ============================================================================
# INITIALIZE MODULES
# ============================================================================
try:
    if st.session_state.emotion_detector is None:
        st.session_state.emotion_detector = EmotionDetector(model_path=emotion_model_path)
    
    if st.session_state.spotify_recommender is None:
        st.session_state.spotify_recommender = SpotifyRecommender(csv_path=music_data_path)
    
    if st.session_state.recommendation_engine is None:
        st.session_state.recommendation_engine = RecommendationEngine()

except Exception as e:
    st.error(f"❌ Error initializing modules: {str(e)}")
    st.stop()

# ============================================================================
# MAIN INTERFACE TABS
# ============================================================================
tab1, tab2 = st.tabs(["🎥 Live Detection", "📊 Analytics"])

# ============================================================================
# TAB 1: LIVE DETECTION
# ============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📷 Live Webcam Feed")
        st.markdown("Click 'Start Detection' to find your emotion. The stream will stop automatically when an emotion is detected with high confidence.")
        
        # Start/Stop buttons
        if st.button("▶️ Start Detection", key="start_btn"):
            st.session_state.run_webcam = True
            st.session_state.current_emotion = None # Reset emotion on new start
            st.rerun()

        if st.button("⏹️ Stop Detection", key="stop_btn"):
            st.session_state.run_webcam = False
            st.rerun()

        # Placeholder for webcam feed
        webcam_placeholder = st.empty()
        
        if st.session_state.run_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Unable to access webcam. Please check permissions.")
                st.session_state.run_webcam = False
            else:
                st.info("🔎 Looking for a face... The stream will stop when an emotion is detected.")
                while st.session_state.run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to capture frame. Stopping.")
                        st.session_state.run_webcam = False
                        break
                    
                    frame = cv2.flip(frame, 1) # Flip for selfie view
                    
                    # Detect emotion
                    try:
                        emotion, confidence, annotated_frame = st.session_state.emotion_detector.detect(
                            frame,
                            confidence_threshold=0.1 # Use a low threshold to always get a reading
                        )
                        
                        # Display annotated frame
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        webcam_placeholder.image(frame_rgb, use_column_width='always')


                        # Check if a confident emotion was detected
                        if emotion and confidence >= confidence_threshold:
                            st.session_state.current_emotion = {
                                "emotion": emotion,
                                "confidence": confidence,
                                "timestamp": datetime.now()
                            }
                            st.session_state.emotion_history.append(st.session_state.current_emotion)
                            st.session_state.run_webcam = False # Stop the loop
                            st.rerun() # Rerun to update the UI
                        
                    except Exception as e:
                        st.error(f"❌ Error during detection: {str(e)}")
                        st.session_state.run_webcam = False
                        break
                
                cap.release()
        else:
            # Show a static image or a message when the webcam is off
            webcam_placeholder.info("Webcam is off. Click 'Start Detection' to begin.")

    with col2:
        st.markdown("### ✨ Detected Emotion")
        emotion_placeholder = st.empty()
        if st.session_state.current_emotion:
            emotion_data = st.session_state.current_emotion
            emotion_class = f"emotion-{emotion_data['emotion'].lower()}"
            emotion_placeholder.markdown(
                f'<div class="emotion-badge {emotion_class}">😊 Detected: {emotion_data["emotion"].upper()} ({emotion_data["confidence"]:.2%})</div>',
                unsafe_allow_html=True
            )
            st.write(f"🕐 Last detected: {emotion_data['timestamp'].strftime('%H:%M:%S')}")
        else:
            emotion_placeholder.info("No emotion detected yet. Start the webcam to analyze your mood!")

# ============================================================================
# RECOMMENDATIONS SECTION (BELOW TABS)
# ============================================================================
st.markdown("---")
st.markdown("### 🎼 Music Recommendations")

if st.session_state.current_emotion:
    emotion = st.session_state.current_emotion["emotion"]
    
    # Automatically trigger recommendations when a new emotion is detected
    if "last_emotion_for_recs" not in st.session_state or st.session_state.last_emotion_for_recs != emotion:
        with st.spinner("🎵 Fetching recommendations based on your new mood..."):
            try:
                recommendations = st.session_state.recommendation_engine.get_recommendations(
                    emotion=emotion,
                    spotify_recommender=st.session_state.spotify_recommender,
                    num_recommendations=num_recommendations,
                    energy_weight=energy_weight,
                    valence_weight=valence_weight
                )
                st.session_state.current_recommendations = recommendations
                st.session_state.last_emotion_for_recs = emotion # Mark as updated
            except Exception as e:
                st.error(f"❌ Error fetching recommendations: {str(e)}")

    if st.session_state.current_recommendations:
        st.success(f"Here are some **{emotion.title()}** vibes for you!")
        for i, song in enumerate(st.session_state.current_recommendations):
            st.markdown(f"""
            <div class="recommendation-card">
                <h5>{i+1}. 🎶 {song['name']}</h5>
                <p>🎤 {song['artist']} | 💿 {song['album']}</p>
                <p>⚡ Energy: {song['energy']:.2f} | 💃 Danceability: {song['danceability']:.2f} | 😊 Valence: {song['valence']:.2f}</p>
                <a href="{song['external_urls']['spotify']}" target="_blank">Open in Spotify</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Recommendations will appear here once your emotion is detected.")

else:
    st.info("No recommendations yet. Start the live detection to get your personalized playlist!")

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================
with tab2:
    st.markdown("### 📊 Emotion History & Trends")
    if st.session_state.emotion_history:
        history_df = pd.DataFrame(st.session_state.emotion_history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.markdown("#### Emotion Log")
        st.dataframe(history_df[['timestamp', 'emotion', 'confidence']], width=None, use_container_width=True) # Keeping both for compatibility, but the warning should disappear. A more modern way is just use_container_width=True, but let's ensure it works. The warning is just a warning.


        st.markdown("#### Emotion Distribution")
        emotion_counts = history_df['emotion'].value_counts()
        st.bar_chart(emotion_counts)
    else:
        st.info("No emotion history yet. Start detecting emotions to see analytics!")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("Built with ❤️ for music lovers and data scientists")
