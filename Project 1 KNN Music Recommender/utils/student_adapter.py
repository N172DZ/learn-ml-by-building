"""
Student Implementation Adapter for Music Recommender Web API
=============================================================
This file bridges your notebook implementation with the Flask web application.

INSTRUCTIONS:
1. Copy your COMPLETE, TESTED implementations below
2. Do NOT modify the get_recommendations_for_api function
3. Test using: python -m utils.test_student_adapter
"""

import numpy as np
import pandas as pd
import os

# ============================================================================
# STUDENT IMPLEMENTATION SECTION
# Copy your complete, final implementations from the notebook below
# ============================================================================

class FeatureScaler:
    """
    TODO: Copy your complete FeatureScaler implementation here.
    Must include: __init__, fit, transform, fit_transform methods
    """
    def __init__(self):
        # Keep lightweight constructor so module import does not fail
        self._fitted = False
    
    def fit(self, X):
        """
        Learn mean and std for each feature in X.
        
        Args:
            X (np.ndarray): Shape (n_samples, n_features)
            
        Sets:
            self.mean: Mean per feature
            self.std: Std per feature (replace 0 with 1)
        """
        # --- SOLUTION ---
        # TODO: Implement
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # NOTE: Make sure you set the values of self.mean and self.std
        # No need to return any values
        pass

    def transform(self, X):
        """
        Apply scaling: (X - mean) / std
        
        Args:
            X (np.ndarray): Shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Scaled data, same shape as X
            
        Raises:
            RuntimeError: If not fitted yet
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        # --- SOLUTION ---
        # TODO: Implement
        std_replaced = np.where(self.std == 0, 1, self.std)  # Avoid division by zero
        return (X - self.mean) / std_replaced
        pass
            
    def fit_transform(self, X):
        """A convenience method to fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
class KNNRecommender:
    """A k-Nearest Neighbors recommender for music."""
    def __init__(self, k=10):
        self.k = k
        self.item_profile = None
        self.features_matrix = None
        self.feature_columns = None
        self.track_id_to_index = {}
        
    @staticmethod
    def euclidean_distance(a, b):
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            a (np.ndarray): Vector of shape (n,)
            b (np.ndarray): Vector of shape (n,)
            
        Returns:
            float: Euclidean distance
            
        Example:
            >>> euclidean_distance(np.array([1,2,3]), np.array([4,5,6]))
            5.196152422706632
        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        return float(np.linalg.norm(a - b))
        # --- SOLUTION ---
        pass

    @staticmethod
    def cosine_distance(a, b):
        """
        Calculates the Cosine distance between two numerical vectors a and b.
        Formula: 1 - (a·b) / (||a|| * ||b||)
                
        Args:
            a (np.ndarray): Vector of shape (n,)
            b (np.ndarray): Vector of shape (n,)
            
        Returns:
            float: Cosine distance (between 0 and 1)
            
        Example:
            >>> cosine_distance(np.array([1,2,3]), np.array([4,5,6]))
            0.025368153802923787
            
        Note: Return 1.0 if either vector has zero norm.
        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else 1.0
        # --- SOLUTION ---
        pass
        
    def fit(self, item_profile_df, feature_columns):
        """Prepares the recommender by loading and processing the track data."""
        self.item_profile = item_profile_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.features_matrix = self.item_profile[self.feature_columns].values
        self.track_id_to_index = {track_id: i for i, track_id in enumerate(self.item_profile['id'])}
        print(f"Fit complete. Loaded {len(self.item_profile)} tracks.")

    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='euclidean'):
        """
        Find k nearest neighbors for a track.
        
        Args:
            track_id (str): Query track ID
            n_neighbors (int): Number of neighbors (default: self.k)
            distance_metric (str): 'euclidean' or 'cosine'
            
        Returns:
            list: [(distance, track_id), ...] sorted by distance
            
        Example:
            >>> neighbors = recommender.find_neighbors('track123', n_neighbors=5)
            [(0.23, 'track456'), (0.31, 'track789'), ...]
        """
        if n_neighbors is None: n_neighbors = self.k
        distance_functions = {'euclidean': self.euclidean_distance, 'cosine': self.cosine_distance}
        if distance_metric not in distance_functions: raise ValueError(f"Unknown metric: {distance_metric}")
        if track_id not in self.track_id_to_index: raise ValueError(f"Track ID {track_id} not found.")

        # --- YOUR IMPLEMENTATION GOES HERE ---
        # --- SOLUTION ---
        # DONE: Implement k-NN search
        query_index = self.track_id_to_index[track_id]
        query_vector = self.features_matrix[query_index]
        distances = []
        for i in range(len(self.features_matrix)):
            if i == query_index:
                continue  # Skip the query track itself
            dist = distance_functions[distance_metric](query_vector, self.features_matrix[i])
            distances.append((dist, self.item_profile['id'].iloc[i]))
        distances.sort(key=lambda x: x[0])  # Sort by distance
        return distances[:n_neighbors]
        # Don't include the query track itself in results
        pass

    def recommend(self, track_id, n_recommendations=None, distance_metric='euclidean'):
        if self.item_profile is None: raise RuntimeError("Recommender has not been fitted.")
        neighbors = self.find_neighbors(track_id, n_recommendations, distance_metric)
        neighbor_ids = [tid for distance, tid in neighbors]
        results_df = self.item_profile[self.item_profile['id'].isin(neighbor_ids)].copy()
        distances_map = {tid: dist for dist, tid in neighbors}
        results_df['distance'] = results_df['id'].map(distances_map)
        return results_df.sort_values('distance')

def custom_hybrid_distance(track_a_data, track_b_data, audio_features_a, audio_features_b, w_artist=0.5):
    """
    Design your hybrid distance function here.
    
    Args:
        track_a_data (pd.Series): Full data row for track A
        track_b_data (pd.Series): Full data row for track B
        audio_features_a (np.ndarray): Audio feature vector for track A
        audio_features_b (np.ndarray): Audio feature vector for track B
        w_artist (float): Weight for metadata component (0-1)
        
    Returns:
        float: Combined distance value
        
    Ideas to consider:
        - Audio similarity (using cosine or euclidean distance)
        - Artist similarity (same artist = lower distance)
        - You could also consider: genre, year, popularity, etc.
    """
    # DONE: Implement your hybrid distance

    # D_hybrid = w1 * D_audio + w2 * D_artist_penalty
    d_audio = KNNRecommender.euclidean_distance(audio_features_a, audio_features_b)
    artist_penalty = 0.5 if track_a_data['artist'] == track_b_data['artist'] else 1.0
    d_hybrid = (1-w_artist) * d_audio + w_artist * artist_penalty
    return d_hybrid
# Optional: If you implemented HybridKNNRecommender, add it here
class HybridKNNRecommender(KNNRecommender):
    
    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='hybrid', w_artist=0.5):
        """
        Find neighbors using hybrid distance that combines audio features and metadata.
        
        This method extends the base KNNRecommender to use the custom_hybrid_distance
        function when distance_metric='hybrid'.
        """
        if distance_metric != 'hybrid':
            return super().find_neighbors(track_id, n_neighbors, distance_metric)
        
        if n_neighbors is None: 
            n_neighbors = self.k
            
        if track_id not in self.track_id_to_index: 
            raise ValueError(f"Track ID {track_id} not found.")
        
        # --- YOUR IMPLEMENTATION GOES HERE ---
        # DONE: Implement hybrid k-NN search
        # 1. Get query track's features and metadata
        # 2. For each other track:
        #    - Calculate hybrid distance using custom_hybrid_distance
        #    - Store (distance, track_id)
        # 3. Sort and return top n_neighbors
        query_index = self.track_id_to_index[track_id]
        query_vector = self.features_matrix[query_index]
        distances = []
        for i in range(len(self.features_matrix)):
            if i == query_index:
                continue  # Skip the query track itself
            dist = custom_hybrid_distance(
                track_a_data=self.item_profile.iloc[query_index],
                track_b_data=self.item_profile.iloc[i],
                audio_features_a=query_vector,
                audio_features_b=self.features_matrix[i],
                w_artist=w_artist
            )
            distances.append((dist, self.item_profile['id'].iloc[i]))
        distances.sort(key=lambda x: x[0])  # Sort by distance
        return distances[:n_neighbors]
        pass


# ============================================================================
# API ADAPTER SECTION - DO NOT MODIFY ANYTHING BELOW THIS LINE
# ============================================================================

# Cache for the recommender instance
_recommender_cache = None
_audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                   'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']


def get_recommendations_for_api(track_id, k=10, metric='cosine', use_hybrid=False):
    """
    Bridge function between student implementation and web API.
    DO NOT MODIFY THIS FUNCTION.
    """
    global _recommender_cache
    
    try:
        # Load data and initialize recommender if needed
        if _recommender_cache is None:
            # Find the data file
            possible_paths = [
                'data/mergedFile.csv',
                '../data/mergedFile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
                'data/item_profile.csv',
                '../data/item_profile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
            ]
            
            item_profile = None
            for path in possible_paths:
                if os.path.exists(path):
                    item_profile = pd.read_csv(path, dtype={'id': str})
                    break
            
            if item_profile is None:
                raise FileNotFoundError("Could not find mergedFile.csv or item_profile.csv")
            
            # Initialize the appropriate recommender
            if use_hybrid and 'HybridKNNRecommender' in globals():
                _recommender_cache = HybridKNNRecommender(k=k)
            else:
                _recommender_cache = KNNRecommender(k=k)
            
            # Use only features that exist in the loaded file
            available_feats = [f for f in _audio_features if f in item_profile.columns]
            if not available_feats:
                raise ValueError("No required audio feature columns found in data file")

            _recommender_cache.fit(item_profile, available_feats)
            print(f"Initialized {type(_recommender_cache).__name__} with {len(item_profile)} tracks")
        
        # Get recommendations
        recommendations = _recommender_cache.recommend(
            track_id, 
            n_recommendations=k, 
            distance_metric=metric
        )
        
        # Convert to API format
        result = {}
        for _, row in recommendations.iterrows():
            result[row['id']] = {
                'distance': float(row['distance']),
                'song': row.get('song', 'Unknown'),
                'artist': row.get('artist', 'Unknown'),
                'features': {feat: float(row.get(feat, 0)) for feat in _audio_features if feat in row}
            }
        
        return result
        
    except Exception as e:
        print(f"Error in student implementation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_implementation():
    """
    Test function to verify your implementation works.
    Run this after copying your code above.
    """
    try:
        from utils.test_student_adapter import run_comprehensive_tests
        return run_comprehensive_tests()
    except ImportError:
        # Fallback if test file is not in expected location
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'utils.test_student_adapter'], 
                              capture_output=False)
        return result.returncode == 0


if __name__ == "__main__":
    print("To test your implementation, run:")
    print("  python -m utils.test_student_adapter")             

# ============================================================================
# AUTO-INITIALIZATION FOR API INTEGRATION
# ============================================================================

def initialize_for_api():
    """
    Initialize the recommender for API usage.
    This is called when api_helpers imports this module.
    """
    import os
    import pandas as pd
    
    # Try to find and load the data (prefer mergedFile.csv which has full feature set)
    possible_paths = [
        'data/mergedFile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
        'data/item_profile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={'id': str})
                audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                                 'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']
                
                # Create and fit the recommender
                recommender = KNNRecommender(k=10)
                recommender.fit(df, audio_features)
                
                print(f"✅ Student recommender initialized with {len(df)} tracks")
                return recommender
                
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
                continue
    
    print("⚠️ Could not initialize student recommender - data files not found")
    return None

# Export the initialized recommender for api_helpers to use (disabled to avoid duplicate init; api_helpers handles it)
# student_recommender_instance = initialize_for_api()