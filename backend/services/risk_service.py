"""
Risk Calculation Service
Combines physics-based calculations with ML predictions.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.utils import risk_classification, collision_probability
from backend.schemas import TimelinePoint


class RiskCalculationService:
    """
    Service for comprehensive collision risk assessment.
    """
    
    def __init__(self, ml_service, orbit_service):
        """
        Initialize risk service.
        
        Args:
            ml_service: ML prediction service
            orbit_service: Orbit propagation service
        """
        self.ml_service = ml_service
        self.orbit_service = orbit_service
    
    def predict_collision_risk(self, sat1: Dict, sat2: Dict, 
                              start_time: datetime = None) -> Dict:
        """
        Comprehensive collision risk prediction.
        
        Combines:
        - ML model prediction
        - Physics-based calculations
        - Probability assessment
        
        Args:
            sat1, sat2: Satellite data dictionaries
            start_time: Prediction start time
            
        Returns:
            Complete risk assessment
        """
        if start_time is None:
            start_time = datetime.utcnow()
        
        # Get ML prediction
        if self.ml_service.is_model_loaded():
            ml_result = self.ml_service.predict(sat1, sat2, start_time)
            predicted_min_dist = ml_result['predicted_min_distance_km']
        else:
            # Fallback: use physics-based closest approach
            closest = self.orbit_service.find_closest_approach(
                sat1, sat2, start_time
            )
            predicted_min_dist = closest['minimum_distance_km']
            ml_result = {
                'current_distance_km': closest['minimum_distance_km'],
                'relative_velocity_kmps': closest['relative_velocity_kmps']
            }
        
        # Calculate current state
        current_distance = ml_result['current_distance_km']
        rel_velocity = ml_result['relative_velocity_kmps']
        
        # Risk classification
        risk_level = risk_classification(predicted_min_dist, rel_velocity)
        
        # Collision probability
        prob = collision_probability(predicted_min_dist)
        
        return {
            'satellite_a': sat1.get('name', sat1.get('norad_id')),
            'satellite_b': sat2.get('name', sat2.get('norad_id')),
            'predicted_min_distance_km': float(predicted_min_dist),
            'current_distance_km': float(current_distance),
            'relative_velocity_kmps': float(rel_velocity),
            'risk_level': risk_level,
            'collision_probability': float(prob),
            'prediction_horizon_hours': 24,
            'timestamp': start_time,
            'confidence': 0.85 if self.ml_service.is_model_loaded() else 0.70
        }
    
    def generate_risk_timeline(self, sat1: Dict, sat2: Dict,
                              duration_hours: int = 24,
                              interval_hours: int = 1) -> List[TimelinePoint]:
        """
        Generate timeline of risk evolution.
        
        Args:
            sat1, sat2: Satellite data
            duration_hours: Timeline duration
            interval_hours: Time between points
            
        Returns:
            List of TimelinePoint objects
        """
        timeline = []
        start_time = datetime.utcnow()
        
        for hour in range(0, duration_hours, interval_hours):
            prediction_time = start_time + timedelta(hours=hour)
            
            try:
                # Predict for this time
                result = self.predict_collision_risk(sat1, sat2, prediction_time)
                
                point = TimelinePoint(
                    hours_from_now=hour,
                    distance_km=result['predicted_min_distance_km'],
                    risk_level=result['risk_level'],
                    timestamp=prediction_time
                )
                
                timeline.append(point)
                
            except Exception as e:
                print(f"Warning: Failed to generate timeline point at hour {hour}: {e}")
                continue
        
        return timeline
    
    def analyze_maneuver_scenario(self, sat_to_maneuver: Dict, 
                                  target_sat: Dict,
                                  delta_v: List[float],
                                  maneuver_time: datetime) -> Dict:
        """
        Analyze effect of orbital maneuver on collision risk.
        
        Args:
            sat_to_maneuver: Satellite performing maneuver
            target_sat: Satellite to avoid
            delta_v: Velocity change vector [vx, vy, vz] km/s
            maneuver_time: When maneuver occurs
            
        Returns:
            Scenario analysis with before/after comparison
        """
        # Risk before maneuver
        risk_before = self.predict_collision_risk(
            sat_to_maneuver, target_sat, maneuver_time
        )
        
        # Simulate maneuver (simplified - modifies TLE conceptually)
        # In production, would use orbit propagation with delta-v application
        # For now, we'll estimate impact
        
        delta_v_mag = np.linalg.norm(delta_v)
        
        # Heuristic: maneuver effectiveness based on delta-v and timing
        # Larger delta-v and earlier timing = more effective
        hours_until_maneuver = (maneuver_time - datetime.utcnow()).total_seconds() / 3600
        effectiveness_factor = min(delta_v_mag * 1000, 100) * (1 + hours_until_maneuver / 24)
        
        # Estimate new minimum distance
        # This is simplified - real calculation would re-propagate orbits
        estimated_new_min_dist = risk_before['predicted_min_distance_km'] + effectiveness_factor
        
        new_risk_level = risk_classification(estimated_new_min_dist, 
                                            risk_before['relative_velocity_kmps'])
        
        # Determine effectiveness
        if new_risk_level == 'SAFE' and risk_before['risk_level'] != 'SAFE':
            effectiveness = "HIGH"
        elif new_risk_level == 'CAUTION' and risk_before['risk_level'] == 'HIGH_RISK':
            effectiveness = "MODERATE"
        else:
            effectiveness = "LOW"
        
        return {
            'satellite': sat_to_maneuver.get('name'),
            'target': target_sat.get('name'),
            'risk_before_maneuver': risk_before['risk_level'],
            'risk_after_maneuver': new_risk_level,
            'min_distance_before_km': risk_before['predicted_min_distance_km'],
            'min_distance_after_km': float(estimated_new_min_dist),
            'maneuver_effectiveness': effectiveness,
            'delta_v_magnitude_mps': float(delta_v_mag * 1000)
        }
    
    def get_high_risk_pairs(self, threshold: str = 'CAUTION', 
                           limit: int = 10) -> List[Dict]:
        """
        Identify high-risk satellite pairs.
        
        Args:
            threshold: Minimum risk level
            limit: Maximum pairs to return
            
        Returns:
            List of high-risk pair assessments
        """
        # This would query database for all pairs and filter
        # For demo, returning placeholder
        # In production, would integrate with satellite database
        
        return [{
            'satellite_a': 'SAT-0001',
            'satellite_b': 'SAT-0002',
            'risk_level': 'HIGH_RISK',
            'predicted_min_distance_km': 4.2,
            'time_of_closest_approach': datetime.utcnow() + timedelta(hours=12)
        }]
