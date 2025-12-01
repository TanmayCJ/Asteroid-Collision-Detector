"""
Prediction module for collision risk assessment.
Provides inference capabilities using trained model.
"""

import numpy as np
from datetime import datetime
from tensorflow import keras

try:
    from ml.preprocessing import OrbitDataPreprocessor
    from ml.utils import tle_to_cartesian, calculate_relative_features
except ImportError:
    from preprocessing import OrbitDataPreprocessor
    from utils import tle_to_cartesian, calculate_relative_features

class CollisionRiskPredictor:
    """Predicts collision risk between two orbiting objects."""
    
    def __init__(self, model_path='models/collision_predictor.h5', scaler_path='models/feature_scaler.pkl'):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to feature scaler file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.preprocessor = None
        print(f"[OK] CollisionRiskPredictor initializing...")
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler from disk."""
        try:
            # Try loading .keras format first (newer)
            keras_path = self.model_path.replace('.h5', '.keras')
            try:
                self.model = keras.models.load_model(keras_path, compile=False)
                print(f"[OK] Model loaded from {keras_path}")
            except:
                # Fall back to H5 format
                self.model = keras.models.load_model(self.model_path, compile=False)
                print(f"[OK] Model loaded from {self.model_path}")
            
            # Recompile model with current Keras
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Initialize preprocessor with same params as training
            self.preprocessor = OrbitDataPreprocessor()
            try:
                self.preprocessor.load_scaler()
            except FileNotFoundError:
                print("[WARNING] Scaler not found, will use default normalization")
            print("[OK] Preprocessor initialized")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def predict_collision_risk(self, tle1_line1, tle1_line2, tle2_line1, tle2_line2):
        """
        Predict collision risk between two objects.
        
        Args:
            tle1_line1, tle1_line2: TLE for first object
            tle2_line1, tle2_line2: TLE for second object
            
        Returns:
            Dictionary with prediction results:
                - risk_probability: Float between 0 and 1
                - risk_level: String ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
                - confidence: Model confidence score
        """
        print("\n[OK] Predicting collision risk...")
        
        try:
            # Propagate orbit pair
            epoch = datetime.now()
            features = self.preprocessor.propagate_orbit_pair(
                tle1_line1, tle1_line2,
                tle2_line1, tle2_line2,
                epoch
            )
            
            if features is None:
                print("[ERROR] Failed to propagate orbits")
                return None
            
            print(f"[OK] Generated features shape: {features.shape}")
            
            # Create sequences for prediction (use first 20 time steps)
            sequence_length = 20
            if features.shape[0] < sequence_length:
                print(f"[ERROR] Not enough time steps. Need {sequence_length}, got {features.shape[0]}")
                return None
            
            # Extract sequence
            sequence = features[:sequence_length]
            
            # Reshape for model input: (1, sequence_length, n_features)
            X = sequence.reshape(1, sequence_length, -1)
            print(f"[OK] Input shape for model: {X.shape}")
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            print(f"[OK] Raw prediction: {prediction:.4f}")
            
            # Determine risk level
            if prediction < 0.25:
                risk_level = 'LOW'
            elif prediction < 0.50:
                risk_level = 'MEDIUM'
            elif prediction < 0.75:
                risk_level = 'HIGH'
            else:
                risk_level = 'CRITICAL'
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(prediction - 0.5) * 2
            
            result = {
                'risk_probability': float(prediction),
                'risk_level': risk_level,
                'confidence': float(confidence)
            }
            
            print(f"[OK] Risk Level: {risk_level}")
            print(f"[OK] Probability: {prediction:.4f}")
            print(f"[OK] Confidence: {confidence:.4f}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None
    
    def predict_minimum_distance(self, sat1_tle: dict, sat2_tle: dict, start_time=None):
        """
        Predict minimum distance between two satellites in next 24 hours.
        This is the main method called by the backend API.
        
        Args:
            sat1_tle: Dict with 'tle_line1' and 'tle_line2' for first satellite
            sat2_tle: Dict with 'tle_line1' and 'tle_line2' for second satellite
            start_time: Optional datetime for prediction start (default: now)
            
        Returns:
            Dictionary with predicted distance and risk assessment
        """
        try:
            if start_time is None:
                start_time = datetime.now()
            
            # Build sat dictionaries
            sat1 = {
                'tle_line1': sat1_tle.get('tle_line1'),
                'tle_line2': sat1_tle.get('tle_line2')
            }
            sat2 = {
                'tle_line1': sat2_tle.get('tle_line1'),
                'tle_line2': sat2_tle.get('tle_line2')
            }
            
            # Propagate orbits and get features
            df = self.preprocessor.propagate_orbit_pair(
                sat1, sat2, start_time, duration_hours=26
            )
            
            if df is None or df.empty:
                raise ValueError("Failed to propagate orbits")
            
            # Get current state (first timestep)
            current_distance = df['distance'].iloc[0]
            current_velocity = df['relative_velocity'].iloc[0]
            
            # Prepare sequence for prediction
            sequence_length = min(12, len(df))
            
            # Add the 12th feature: distance/velocity ratio (time to collision metric)
            df['dist_vel_ratio'] = df['distance'] / (df['relative_velocity'] + 1e-6)
            
            features = df[['distance', 'relative_velocity', 'approach_rate',
                          'pos1_x', 'pos1_y', 'pos1_z', 'pos2_x', 'pos2_y', 'pos2_z',
                          'vel1_mag', 'vel2_mag', 'dist_vel_ratio']].iloc[:sequence_length].values
            
            # Normalize features
            if self.preprocessor.feature_mean is not None:
                features = (features - self.preprocessor.feature_mean) / (self.preprocessor.feature_std + 1e-8)
            
            # Pad if needed
            if len(features) < 12:
                padding = np.zeros((12 - len(features), features.shape[1]))
                features = np.vstack([features, padding])
            
            # Reshape for model: (1, 12, n_features)
            X = features.reshape(1, 12, -1)
            
            # Predict
            predicted_distance = self.model.predict(X, verbose=0)[0][0]
            
            # Classify risk
            if predicted_distance < 5:
                risk_level = 'HIGH_RISK'
            elif predicted_distance < 25:
                risk_level = 'CAUTION'
            else:
                risk_level = 'SAFE'
            
            return {
                'predicted_min_distance_km': float(predicted_distance),
                'current_distance_km': float(current_distance),
                'relative_velocity_kmps': float(current_velocity),
                'risk_level': risk_level,
                'prediction_horizon_hours': 24
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, tle_pairs):
        """
        Predict collision risk for multiple object pairs.
        
        Args:
            tle_pairs: List of dicts with keys:
                - 'object1': dict with 'tle_line1', 'tle_line2'
                - 'object2': dict with 'tle_line1', 'tle_line2'
                
        Returns:
            List of prediction results
        """
        print(f"\n[OK] Batch prediction for {len(tle_pairs)} pairs...")
        
        results = []
        
        for idx, pair in enumerate(tle_pairs):
            print(f"\n[OK] Processing pair {idx + 1}/{len(tle_pairs)}")
            
            obj1 = pair['object1']
            obj2 = pair['object2']
            
            result = self.predict_collision_risk(
                obj1['tle_line1'], obj1['tle_line2'],
                obj2['tle_line1'], obj2['tle_line2']
            )
            
            if result is not None:
                result['pair_index'] = idx
                results.append(result)
        
        print(f"\n[OK] Batch prediction complete: {len(results)}/{len(tle_pairs)} successful")
        
        return results
    
    def get_close_approach_analysis(self, tle1_line1, tle1_line2, tle2_line1, tle2_line2):
        """
        Analyze close approach between two objects over time.
        
        Args:
            tle1_line1, tle1_line2: TLE for first object
            tle2_line1, tle2_line2: TLE for second object
            
        Returns:
            Dictionary with analysis results including minimum distance
        """
        print("\n[OK] Performing close approach analysis...")
        
        try:
            # Propagate orbit pair
            epoch = datetime.now()
            features = self.preprocessor.propagate_orbit_pair(
                tle1_line1, tle1_line2,
                tle2_line1, tle2_line2,
                epoch
            )
            
            if features is None:
                print("[ERROR] Failed to propagate orbits")
                return None
            
            # Extract distances (assuming first feature is relative distance)
            distances = features[:, 0]
            
            # Find minimum distance
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            time_of_closest_approach = min_distance_idx * self.preprocessor.time_step_hours
            
            # Calculate average distance
            avg_distance = np.mean(distances)
            
            analysis = {
                'min_distance_km': float(min_distance),
                'time_of_closest_approach_hours': float(time_of_closest_approach),
                'avg_distance_km': float(avg_distance),
                'all_distances': distances.tolist()
            }
            
            print(f"[OK] Minimum distance: {min_distance:.2f} km at t+{time_of_closest_approach:.1f}h")
            print(f"[OK] Average distance: {avg_distance:.2f} km")
            
            return analysis
            
        except Exception as e:
            print(f"[ERROR] Close approach analysis failed: {e}")
            return None

def main():
    """Example usage of collision risk predictor."""
    print("=" * 60)
    print("COLLISION RISK PREDICTION EXAMPLE")
    print("=" * 60)
    
    # Example TLE data (ISS-like orbit)
    tle1_line1 = "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990"
    tle1_line2 = "2 25544  51.6461 339.8014 0002571  85.5305  34.3968 15.48919393265091"
    
    # Second object with slightly different orbit
    tle2_line1 = "1 12345U 98067B   21001.00000000  .00002182  00000-0  41420-4 0  9991"
    tle2_line2 = "2 12345  51.6500 339.8000 0002600  85.5000  34.4000 15.48919393265092"
    
    # Initialize predictor
    predictor = CollisionRiskPredictor()
    
    # Make prediction
    result = predictor.predict_collision_risk(
        tle1_line1, tle1_line2,
        tle2_line1, tle2_line2
    )
    
    if result:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['risk_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    # Perform close approach analysis
    analysis = predictor.get_close_approach_analysis(
        tle1_line1, tle1_line2,
        tle2_line1, tle2_line2
    )
    
    if analysis:
        print("\n" + "=" * 60)
        print("CLOSE APPROACH ANALYSIS")
        print("=" * 60)
        print(f"Minimum Distance: {analysis['min_distance_km']:.2f} km")
        print(f"Time of Closest Approach: {analysis['time_of_closest_approach_hours']:.1f} hours")
        print(f"Average Distance: {analysis['avg_distance_km']:.2f} km")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
