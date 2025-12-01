"""
AstroGuard Orbit Data Preprocessing
====================================
Handles TLE data processing, feature engineering, and sequence preparation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pickle
import json
from pathlib import Path

try:
    from ml.utils import (
        tle_to_cartesian,
        calculate_distance,
        calculate_relative_velocity,
        calculate_approach_rate,
        generate_synthetic_tle,
        normalize_features
    )
except ImportError:
    from utils import (
        tle_to_cartesian,
        calculate_distance,
        calculate_relative_velocity,
        calculate_approach_rate,
        generate_synthetic_tle,
        normalize_features
    )


class OrbitDataPreprocessor:
    """Preprocessing pipeline for orbit collision data."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'sample_interval_seconds': 60,  # 1 minute
            'prediction_horizon_hours': 24,  # 24 hours ahead
            'sequence_length': 12,  # 12 timesteps in sequence
        }
        self.feature_mean = None
        self.feature_std = None
        self.scaler_path = Path('models/feature_scaler.pkl')
    
    def generate_synthetic_dataset(self, num_satellites: int = 40) -> pd.DataFrame:
        """
        Generate synthetic satellite TLE data for training.
        
        Args:
            num_satellites: Number of satellites to generate
            
        Returns:
            DataFrame with synthetic satellite data
        """
        print(f"Generating {num_satellites} synthetic satellites...")
        
        satellites = []
        np.random.seed(42)
        
        for i in range(num_satellites):
            # Realistic orbital parameters
            semi_major_axis = np.random.uniform(6800, 8000)  # LEO orbits
            eccentricity = np.random.uniform(0.0001, 0.02)   # Nearly circular
            inclination = np.random.uniform(0, 98)            # Various inclinations
            
            tle_line1, tle_line2 = generate_synthetic_tle(
                semi_major_axis, eccentricity, inclination, f"SAT-{i:03d}"
            )
            
            satellites.append({
                'name': f"SAT-{i:03d}",
                'tle_line1': tle_line1,
                'tle_line2': tle_line2,
                'semi_major_axis': semi_major_axis,
                'eccentricity': eccentricity,
                'inclination': inclination
            })
        
        df = pd.DataFrame(satellites)
        
        # Save to file
        output_path = Path('ml/data/synthetic_tle.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(satellites, f, indent=2)
        
        print(f"[OK] Saved synthetic data to {output_path}")
        return df
    
    def propagate_orbit_pair(self, sat1: Dict, sat2: Dict, start_time: datetime, 
                            duration_hours: float = 26) -> pd.DataFrame:
        """
        Propagate two satellites and compute relative features over time.
        Includes data cleaning and outlier filtering.
        
        Args:
            sat1, sat2: Satellite dictionaries with TLE data
            start_time: Start time for propagation
            duration_hours: How long to propagate (hours)
            
        Returns:
            DataFrame with timesteps and features
        """
        interval = timedelta(seconds=self.config['sample_interval_seconds'])
        num_steps = int(duration_hours * 3600 / self.config['sample_interval_seconds'])
        
        timesteps = []
        features_list = []
        
        for step in range(num_steps):
            current_time = start_time + interval * step
            
            # Propagate both satellites
            result1 = tle_to_cartesian(sat1['tle_line1'], sat1['tle_line2'], current_time)
            result2 = tle_to_cartesian(sat2['tle_line1'], sat2['tle_line2'], current_time)
            
            if result1 is None or result2 is None:
                continue
            
            pos1, vel1 = result1
            pos2, vel2 = result2
            
            # Calculate features
            distance = calculate_distance(pos1, pos2)
            rel_velocity = calculate_relative_velocity(vel1, vel2)
            approach_rate = calculate_approach_rate(pos1, pos2, vel1, vel2)
            
            # Data quality checks - filter unrealistic values
            if distance > 50000 or distance < 0.1:  # Filter extreme distances
                continue
            if rel_velocity > 20 or rel_velocity < 0:  # Filter unrealistic velocities (>20 km/s)
                continue
            
            timesteps.append(current_time)
            # Calculate 12th feature: distance/velocity ratio (time-to-collision indicator)
            dist_vel_ratio = distance / (rel_velocity + 1e-6)
            
            features_list.append({
                'distance': distance,
                'relative_velocity': rel_velocity,
                'approach_rate': approach_rate,
                'pos1_x': pos1[0], 'pos1_y': pos1[1], 'pos1_z': pos1[2],
                'pos2_x': pos2[0], 'pos2_y': pos2[1], 'pos2_z': pos2[2],
                'vel1_mag': np.linalg.norm(vel1),
                'vel2_mag': np.linalg.norm(vel2),
                'dist_vel_ratio': dist_vel_ratio
            })
        
        df = pd.DataFrame(features_list)
        df['timestamp'] = timesteps
        
        # Additional cleaning: remove outliers using IQR method
        if len(df) > 0:
            Q1 = df['distance'].quantile(0.25)
            Q3 = df['distance'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature array from propagated orbit data.
        Applies smoothing to reduce noise.
        
        Args:
            df: DataFrame with orbit features
            
        Returns:
            Feature array (num_timesteps, num_features)
        """
        # Apply moving average smoothing to reduce noise
        window_size = 3
        df_smooth = df.copy()
        for col in ['distance', 'relative_velocity', 'approach_rate', 'vel1_mag', 'vel2_mag']:
            df_smooth[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Select feature columns with engineered features
        feature_cols = [
            'distance', 'relative_velocity', 'approach_rate',
            'pos1_x', 'pos1_y', 'pos1_z',
            'pos2_x', 'pos2_y', 'pos2_z',
            'vel1_mag', 'vel2_mag', 'distance'  # distance twice for 12 features
        ]
        
        features = df_smooth[feature_cols].values
        return features
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            features: Feature array (num_timesteps, num_features)
            targets: Target array (num_timesteps,)
            
        Returns:
            X (num_sequences, sequence_length, num_features)
            y (num_sequences,)
        """
        seq_length = self.config['sequence_length']
        
        if len(features) < seq_length:
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(targets[i+seq_length-1])
        
        return np.array(X), np.array(y)
    
    def calculate_future_minimum_distance(self, df: pd.DataFrame, current_idx: int, 
                                         horizon_steps: int) -> float:
        """
        Calculate minimum distance in future horizon.
        
        Args:
            df: DataFrame with distance column
            current_idx: Current timestep index
            horizon_steps: Number of steps to look ahead
            
        Returns:
            Minimum distance in km
        """
        end_idx = min(current_idx + horizon_steps, len(df))
        future_distances = df['distance'].iloc[current_idx:end_idx]
        return future_distances.min() if len(future_distances) > 0 else df['distance'].iloc[current_idx]
    
    def prepare_training_data(self, satellite_pairs: List[Tuple[Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from satellite pairs.
        
        Args:
            satellite_pairs: List of (sat1, sat2) tuples
            
        Returns:
            X (num_samples, sequence_length, num_features)
            y (num_samples,) - minimum distance in prediction horizon
        """
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        all_X, all_y = [], []
        
        print(f"Processing {len(satellite_pairs)} satellite pairs...")
        
        for idx, (sat1, sat2) in enumerate(satellite_pairs):
            if idx % 10 == 0:
                print(f"  Processing pair {idx}/{len(satellite_pairs)}")
            
            # Propagate orbit pair
            try:
                df = self.propagate_orbit_pair(
                    sat1, sat2, 
                    start_time, 
                    duration_hours=self.config['prediction_horizon_hours'] + 2
                )
                print(f"  [DEBUG] Pair {idx}: propagated {len(df)} timesteps")
            except Exception as e:
                print(f"  [DEBUG] Pair {idx}: propagation failed - {e}")
                continue
            
            if len(df) < self.config['sequence_length'] + 10:
                print(f"  [DEBUG] Skipping pair {idx}: insufficient data (got {len(df)}, need {self.config['sequence_length'] + 10})")
                continue
            
            # Extract features
            try:
                features = self.extract_features(df)
                print(f"  [DEBUG] Pair {idx}: extracted features shape {features.shape}")
            except Exception as e:
                print(f"  [DEBUG] Pair {idx}: feature extraction failed - {e}")
                continue
            
            # Calculate target for each timestep
            horizon_steps = int(self.config['prediction_horizon_hours'] * 3600 / 
                              self.config['sample_interval_seconds'])
            print(f"  [DEBUG] Pair {idx}: calculating targets with horizon_steps={horizon_steps}")
            
            targets = []
            for i in range(len(df) - horizon_steps):
                min_dist = self.calculate_future_minimum_distance(df, i, horizon_steps)
                targets.append(min_dist)
            
            print(f"  [DEBUG] Pair {idx}: calculated {len(targets)} target values")
            
            # Trim features to match targets
            features = features[:len(targets)]
            targets = np.array(targets)
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(features, targets)
            print(f"  [DEBUG] Pair {idx}: created {len(X_seq)} sequences")
            
            if len(X_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
            else:
                print(f"  [DEBUG] Pair {idx}: no sequences created, skipping")
        
        if len(all_X) == 0:
            print("\n[ERROR] No valid sequences were created!")
            return np.array([]), np.array([])
        
        # Concatenate all pairs
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Normalize features
        X_flat = X.reshape(-1, X.shape[-1])
        X_normalized, self.feature_mean, self.feature_std = normalize_features(X_flat)
        X = X_normalized.reshape(X.shape)
        
        print(f"\n[OK] Prepared {len(X)} training samples")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Target range: {y.min():.2f} - {y.max():.2f} km")
        
        return X, y
    
    def save_scaler(self):
        """Save normalization parameters for inference."""
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({
                'mean': self.feature_mean,
                'std': self.feature_std
            }, f)
        print(f"[OK] Saved feature scaler to {self.scaler_path}")
    
    def load_scaler(self):
        """Load normalization parameters."""
        with open(self.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.feature_mean = scaler['mean']
        self.feature_std = scaler['std']
        print(f"[OK] Loaded feature scaler from {self.scaler_path}")


def create_satellite_pairs(satellites: pd.DataFrame, 
                          max_pairs: int = 100) -> List[Tuple[Dict, Dict]]:
    """
    Create pairs of satellites for collision analysis.
    Prioritizes satellites in similar orbits (more likely to have close approaches).
    
    Args:
        satellites: DataFrame with satellite TLE data
        max_pairs: Maximum number of pairs to generate
        
    Returns:
        List of (satellite1, satellite2) tuples
    """
    pairs = []
    satellites_list = satellites.to_dict('records')
    
    # Create pairs with similar orbits
    for i in range(len(satellites_list)):
        for j in range(i+1, len(satellites_list)):
            if len(pairs) >= max_pairs:
                break
            
            sat1 = satellites_list[i]
            sat2 = satellites_list[j]
            
            # Check if orbits are similar (more likely to have close approaches)
            alt_diff = abs(sat1['semi_major_axis'] - sat2['semi_major_axis'])
            inc_diff = abs(sat1['inclination'] - sat2['inclination'])
            
            # Favor pairs with similar altitude and inclination
            if alt_diff < 500 and inc_diff < 30:
                pairs.append((sat1, sat2))
        
        if len(pairs) >= max_pairs:
            break
    
    # Fill remaining with random pairs if needed
    while len(pairs) < max_pairs and len(satellites_list) > 1:
        i = np.random.randint(0, len(satellites_list))
        j = np.random.randint(0, len(satellites_list))
        if i != j:
            pairs.append((satellites_list[i], satellites_list[j]))
    
    print(f"\n[OK] Created {len(pairs)} satellite pairs for analysis")
    return pairs
