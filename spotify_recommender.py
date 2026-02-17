"""
Local Music Recommender Module (Replaces Spotify API)

Handles music recommendations by querying a local CSV dataset
containing track metadata and audio features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

class SpotifyRecommender:
    """
    Recommendation engine using a local CSV dataset.
    Maintains the same interface as the original Spotify module for compatibility.
    """
    
    # Emotion to music parameter mapping
    EMOTION_TO_PARAMS = {
        'Happy': {'energy_range': (0.6, 1.0), 'danceability_range': (0.6, 1.0), 'genres': ['Pop', 'Dance', 'Electronic']},
        'Sad': {'energy_range': (0.0, 0.4), 'danceability_range': (0.0, 0.4), 'genres': ['Indie', 'Acoustic', 'Soul']},
        'Angry': {'energy_range': (0.7, 1.0), 'danceability_range': (0.4, 0.8), 'genres': ['Metal', 'Rock']},
        'Neutral': {'energy_range': (0.4, 0.7), 'danceability_range': (0.4, 0.7), 'genres': ['Pop', 'Indie']},
        'Surprised': {'energy_range': (0.6, 0.9), 'danceability_range': (0.5, 0.9), 'genres': ['Electronic', 'Pop']},
        'Fearful': {'energy_range': (0.1, 0.5), 'danceability_range': (0.1, 0.4), 'genres': ['Ambient', 'Electronic']},
        'Disgusted': {'energy_range': (0.5, 0.8), 'danceability_range': (0.2, 0.6), 'genres': ['Rock', 'Metal']}
    }

    def __init__(self, client_id=None, client_secret=None, csv_path="data/music_data.csv"):
        """
        Initialize the recommender by loading the local dataset.
        """
        # We ignore client_id/secret now as we use local data
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.df)} tracks from local dataset.")
        else:
            print(f"⚠️ Warning: Dataset not found at {csv_path}. Creating mock data.")
            self.df = self._create_mock_df()

    def _create_mock_df(self):
        """Creates a small mock dataframe if the CSV is missing."""
        return pd.DataFrame({
            'track_id': ['mock1'], 'track_name': ['No Data Found'], 
            'artist_name': ['System'], 'album_name': ['None'],
            'genre': ['Pop'], 'energy': [0.5], 'danceability': [0.5],
            'popularity': [50]
        })

    def get_recommendations(
        self,
        emotion: str,
        limit: int = 10,
        energy_weight: float = 0.5,
        valence_weight: float = 0.5 # Kept for signature compatibility
    ) -> List[Dict]:
        """
        Get song recommendations from the local CSV based on emotion.
        """
        if emotion not in self.EMOTION_TO_PARAMS:
            emotion = 'Neutral'
            
        params = self.EMOTION_TO_PARAMS[emotion]
        
        # Filter by genre first
        mask = self.df['genre'].isin(params['genres'])
        filtered_df = self.df[mask].copy()
        
        # If no matches in genre, use whole dataset
        if filtered_df.empty:
            filtered_df = self.df.copy()
            
        # Calculate a simple distance score to target parameters
        target_energy = params['energy_range'][0] + (params['energy_range'][1] - params['energy_range'][0]) * energy_weight
        
        # Score based on energy proximity and popularity
        filtered_df['score'] = 1 - abs(filtered_df['energy'] - target_energy)
        filtered_df['score'] = (filtered_df['score'] * 0.7) + (filtered_df['popularity'] / 100.0 * 0.3)
        
        # Get top results
        results = filtered_df.sort_values(by='score', ascending=False).head(limit)
        
        # Convert to the format expected by app.py
        songs = []
        for _, row in results.iterrows():
            songs.append({
                'id': row['track_id'],
                'name': row['track_name'],
                'artist': row['artist_name'],
                'album': row['album_name'],
                'energy': row['energy'],
                'danceability': row['danceability'],
                'valence': row.get('valence', 0.5), # CSV might not have valence, default to 0.5
                'popularity': row['popularity'],
                'external_urls': {'spotify': f"spotify:search:{row['track_name']} {row['artist_name']}"}
            })
            
        return songs

    def current_user(self):
        """Mock method for compatibility."""
        return {"display_name": "Local User"}
