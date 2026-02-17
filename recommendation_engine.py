"""
Recommendation Engine Module

Combines emotion detection with Spotify audio features to generate
personalized music recommendations using content-based and collaborative filtering.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from datetime import datetime

class RecommendationEngine:
    """
    Advanced recommendation engine combining emotion-based and content-based filtering.
    
    Attributes:
        user_history (list): History of user interactions
        emotion_to_audio_mapping (dict): Maps emotions to audio feature targets
    """
    
    # Mapping of emotions to target audio features
    EMOTION_TO_AUDIO_MAPPING = {
        'Happy': {
            'energy': 0.8,
            'valence': 0.9,
            'danceability': 0.7,
            'acousticness': 0.3,
            'instrumentalness': 0.2
        },
        'Sad': {
            'energy': 0.3,
            'valence': 0.2,
            'danceability': 0.3,
            'acousticness': 0.6,
            'instrumentalness': 0.4
        },
        'Angry': {
            'energy': 0.9,
            'valence': 0.3,
            'danceability': 0.6,
            'acousticness': 0.2,
            'instrumentalness': 0.1
        },
        'Neutral': {
            'energy': 0.5,
            'valence': 0.5,
            'danceability': 0.5,
            'acousticness': 0.4,
            'instrumentalness': 0.3
        },
        'Surprised': {
            'energy': 0.7,
            'valence': 0.7,
            'danceability': 0.6,
            'acousticness': 0.3,
            'instrumentalness': 0.2
        },
        'Fearful': {
            'energy': 0.4,
            'valence': 0.3,
            'danceability': 0.2,
            'acousticness': 0.5,
            'instrumentalness': 0.6
        },
        'Disgusted': {
            'energy': 0.7,
            'valence': 0.2,
            'danceability': 0.4,
            'acousticness': 0.2,
            'instrumentalness': 0.3
        }
    }
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.user_history = []
        self.liked_songs = []
        self.disliked_songs = []
    
    def get_recommendations(
        self,
        emotion: str,
        spotify_recommender,
        num_recommendations: int = 10,
        energy_weight: float = 0.5,
        valence_weight: float = 0.5,
        use_history: bool = True
    ) -> List[Dict]:
        """
        Generate personalized recommendations based on emotion and user history.
        
        Args:
            emotion (str): Detected emotion
            spotify_recommender: SpotifyRecommender instance
            num_recommendations (int): Number of recommendations to return
            energy_weight (float): Weight for energy (0-1)
            valence_weight (float): Weight for valence (0-1)
            use_history (bool): Whether to use user history for filtering
        
        Returns:
            List[Dict]: Ranked list of recommended songs
        """
        try:
            # Get base recommendations from Spotify
            recommendations = spotify_recommender.get_recommendations(
                emotion=emotion,
                limit=num_recommendations * 2,  # Get extra to filter
                energy_weight=energy_weight,
                valence_weight=valence_weight
            )
            
            # Rank and filter recommendations
            ranked_recommendations = self._rank_recommendations(
                recommendations,
                emotion,
                use_history=use_history
            )
            
            # Return top N
            return ranked_recommendations[:num_recommendations]
        
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []
    
    def _rank_recommendations(
        self,
        songs: List[Dict],
        emotion: str,
        use_history: bool = True
    ) -> List[Dict]:
        """
        Rank recommendations based on emotion alignment and user history.
        
        Args:
            songs (List[Dict]): List of candidate songs
            emotion (str): Detected emotion
            use_history (bool): Whether to consider user history
        
        Returns:
            List[Dict]: Ranked songs with scores
        """
        if not songs:
            return []
        
        # Get target audio features for emotion
        target_features = self.EMOTION_TO_AUDIO_MAPPING.get(emotion, {})
        
        # Calculate emotion alignment scores
        for song in songs:
            emotion_score = self._calculate_emotion_alignment(song, target_features)
            popularity_score = song.get('popularity', 0) / 100.0
            
            # Combined score
            song['recommendation_score'] = (emotion_score * 0.7) + (popularity_score * 0.3)
            
            # Apply history-based filtering
            if use_history:
                if song['id'] in [s['id'] for s in self.disliked_songs]:
                    song['recommendation_score'] *= 0.5  # Penalize disliked songs
                elif song['id'] in [s['id'] for s in self.liked_songs]:
                    song['recommendation_score'] *= 1.2  # Boost liked songs
        
        # Sort by recommendation score
        ranked = sorted(songs, key=lambda x: x['recommendation_score'], reverse=True)
        
        return ranked
    
    def _calculate_emotion_alignment(self, song: Dict, target_features: Dict) -> float:
        """
        Calculate how well a song aligns with target emotion features.
        
        Args:
            song (Dict): Song with audio features
            target_features (Dict): Target audio features for emotion
        
        Returns:
            float: Alignment score (0-1)
        """
        if not target_features:
            return 0.5
        
        # Extract audio features from song
        song_features = []
        target_values = []
        
        for feature_name, target_value in target_features.items():
            if feature_name in song:
                song_features.append(song[feature_name])
                target_values.append(target_value)
        
        if not song_features:
            return 0.5
        
        # Calculate cosine similarity
        song_array = np.array(song_features).reshape(1, -1)
        target_array = np.array(target_values).reshape(1, -1)
        
        similarity = cosine_similarity(song_array, target_array)[0][0]
        
        # Normalize to 0-1 range
        alignment_score = (similarity + 1) / 2
        
        return alignment_score
    
    def add_to_history(self, song: Dict, emotion: str, rating: Optional[int] = None):
        """
        Add a song interaction to user history.
        
        Args:
            song (Dict): Song data
            emotion (str): Emotion when song was recommended
            rating (int, optional): User rating (1-5)
        """
        interaction = {
            'song_id': song['id'],
            'song_name': song['name'],
            'emotion': emotion,
            'timestamp': datetime.now(),
            'rating': rating
        }
        
        self.user_history.append(interaction)
        
        # Update liked/disliked lists
        if rating and rating >= 4:
            self.liked_songs.append(song)
        elif rating and rating <= 2:
            self.disliked_songs.append(song)
    
    def get_user_profile(self) -> Dict:
        """
        Get user preference profile based on history.
        
        Returns:
            Dict: User profile with emotion preferences and audio feature preferences
        """
        if not self.user_history:
            return {}
        
        history_df = pd.DataFrame(self.user_history)
        
        profile = {
            'total_interactions': len(self.user_history),
            'favorite_emotions': history_df['emotion'].value_counts().to_dict(),
            'average_rating': history_df['rating'].mean() if 'rating' in history_df else None,
            'liked_songs_count': len(self.liked_songs),
            'disliked_songs_count': len(self.disliked_songs)
        }
        
        return profile
    
    def get_emotion_based_playlist(
        self,
        emotion: str,
        spotify_recommender,
        duration_minutes: int = 30
    ) -> List[Dict]:
        """
        Generate a playlist for a specific emotion with target duration.
        
        Args:
            emotion (str): Target emotion
            spotify_recommender: SpotifyRecommender instance
            duration_minutes (int): Target playlist duration in minutes
        
        Returns:
            List[Dict]: Playlist songs
        """
        avg_song_duration_ms = 3.5 * 60 * 1000  # ~3.5 minutes average
        target_songs = int((duration_minutes * 60 * 1000) / avg_song_duration_ms)
        
        recommendations = self.get_recommendations(
            emotion=emotion,
            spotify_recommender=spotify_recommender,
            num_recommendations=target_songs,
            use_history=True
        )
        
        return recommendations
    
    def get_emotion_transition_playlist(
        self,
        start_emotion: str,
        end_emotion: str,
        spotify_recommender,
        num_songs: int = 20
    ) -> List[Dict]:
        """
        Generate a playlist that transitions between two emotions.
        
        Args:
            start_emotion (str): Starting emotion
            end_emotion (str): Ending emotion
            spotify_recommender: SpotifyRecommender instance
            num_songs (int): Number of songs in playlist
        
        Returns:
            List[Dict]: Transition playlist
        """
        playlist = []
        
        # Get recommendations for both emotions
        start_recs = spotify_recommender.get_recommendations(
            emotion=start_emotion,
            limit=num_songs // 2
        )
        
        end_recs = spotify_recommender.get_recommendations(
            emotion=end_emotion,
            limit=num_songs // 2
        )
        
        # Interleave for smooth transition
        for i in range(len(start_recs)):
            playlist.append(start_recs[i])
            if i < len(end_recs):
                playlist.append(end_recs[i])
        
        return playlist[:num_songs]
    
    def get_similar_songs(
        self,
        song: Dict,
        spotify_recommender,
        num_recommendations: int = 10
    ) -> List[Dict]:
        """
        Get songs similar to a given song.
        
        Args:
            song (Dict): Reference song
            spotify_recommender: SpotifyRecommender instance
            num_recommendations (int): Number of similar songs to return
        
        Returns:
            List[Dict]: Similar songs
        """
        similar = spotify_recommender.get_similar_tracks(
            track_id=song['id'],
            limit=num_recommendations
        )
        
        return similar
    
    def filter_by_audio_features(
        self,
        songs: List[Dict],
        feature_ranges: Dict
    ) -> List[Dict]:
        """
        Filter songs by audio feature ranges.
        
        Args:
            songs (List[Dict]): List of songs to filter
            feature_ranges (Dict): Feature ranges {feature: (min, max)}
        
        Returns:
            List[Dict]: Filtered songs
        """
        filtered = []
        
        for song in songs:
            include = True
            
            for feature, (min_val, max_val) in feature_ranges.items():
                if feature in song:
                    if not (min_val <= song[feature] <= max_val):
                        include = False
                        break
            
            if include:
                filtered.append(song)
        
        return filtered
