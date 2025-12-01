"""
ML Prediction Service
Handles ML model loading and inference for collision prediction.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.predict import CollisionRiskPredictor


class MLPredictionService:
    """
    Service for ML-based collision predictions.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize ML service.
        
        Args:
            model_path: Path to trained model (optional)
            scaler_path: Path to feature scaler (optional)
        """
        if model_path is None:
            model_path = "ml/models/collision_predictor.h5"
        if scaler_path is None:
            scaler_path = "ml/models/feature_scaler.pkl"
        
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.predictor = None
        self._model_loaded = False
    
    def load_model(self):
        """Load trained ML model."""
        try:
            self.predictor = CollisionRiskPredictor(
                str(self.model_path),
                str(self.scaler_path)
            )
            self._model_loaded = True
            print(f"✓ ML model loaded successfully")
        except FileNotFoundError as e:
            print(f"Warning: ML model not found. Run ml/train.py first.")
            print(f"  {e}")
            self._model_loaded = False
        except Exception as e:
            print(f"Error loading ML model: {e}")
            self._model_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded
    
    def predict(self, sat1_tle: Dict, sat2_tle: Dict, 
                start_time=None) -> Dict:
        """
        Predict collision risk using ML model.
        
        Args:
            sat1_tle: TLE data for satellite 1
            sat2_tle: TLE data for satellite 2
            start_time: Prediction start time
            
        Returns:
            Prediction dictionary with risk assessment
        """
        if not self.is_model_loaded():
            raise RuntimeError("ML model not loaded. Cannot make predictions.")
        
        result = self.predictor.predict_minimum_distance(
            sat1_tle, sat2_tle, start_time
        )
        
        return result
    
    def predict_timeline(self, sat1_tle: Dict, sat2_tle: Dict,
                        duration_hours: int = 24,
                        interval_hours: int = 1) -> list:
        """
        Generate timeline of predictions.
        
        Args:
            sat1_tle, sat2_tle: TLE data
            duration_hours: Prediction horizon
            interval_hours: Time between predictions
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_model_loaded():
            raise RuntimeError("ML model not loaded")
        
        num_predictions = duration_hours // interval_hours
        timeline = self.predictor.predict_timeline(
            sat1_tle, sat2_tle,
            num_predictions=num_predictions
        )
        
        return timeline
