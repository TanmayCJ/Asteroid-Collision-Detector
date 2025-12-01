"""
Orbit Propagation Service
Uses SGP4 for satellite orbit calculations.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.utils import (
    tle_to_cartesian,
    calculate_distance,
    calculate_relative_velocity,
    calculate_approach_rate,
    orbital_elements
)


class OrbitPropagationService:
    """
    Service for satellite orbit propagation and calculations.
    """
    
    def __init__(self):
        self.earth_radius_km = 6371.0
    
    def propagate_satellite(self, tle_line1: str, tle_line2: str, 
                           time: datetime) -> Dict:
        """
        Propagate a satellite to a specific time.
        
        Args:
            tle_line1: First TLE line
            tle_line2: Second TLE line
            time: Target time
            
        Returns:
            Dictionary with position, velocity, and orbital elements
        """
        position, velocity = tle_to_cartesian(tle_line1, tle_line2, time)
        
        # Calculate orbital elements
        orb_elements = orbital_elements(position, velocity)
        
        # Calculate altitude
        altitude = np.linalg.norm(position) - self.earth_radius_km
        
        return {
            'timestamp': time,
            'position': position.tolist(),
            'velocity': velocity.tolist(),
            'altitude_km': float(altitude),
            'orbital_elements': orb_elements
        }
    
    def propagate_pair(self, sat1_tle: Dict, sat2_tle: Dict,
                      start_time: datetime,
                      duration_hours: float = 24.0,
                      interval_minutes: int = 10) -> List[Dict]:
        """
        Propagate two satellites and calculate relative metrics.
        
        Args:
            sat1_tle: TLE data for satellite 1
            sat2_tle: TLE data for satellite 2
            start_time: Start time
            duration_hours: How long to propagate
            interval_minutes: Time between samples
            
        Returns:
            List of dictionaries with timestamped data
        """
        results = []
        interval_seconds = interval_minutes * 60
        num_steps = int(duration_hours * 3600 / interval_seconds)
        
        for step in range(num_steps):
            current_time = start_time + timedelta(seconds=step * interval_seconds)
            
            try:
                # Propagate both satellites
                pos1, vel1 = tle_to_cartesian(
                    sat1_tle['tle_line1'],
                    sat1_tle['tle_line2'],
                    current_time
                )
                pos2, vel2 = tle_to_cartesian(
                    sat2_tle['tle_line1'],
                    sat2_tle['tle_line2'],
                    current_time
                )
                
                # Calculate metrics
                distance = calculate_distance(pos1, pos2)
                rel_velocity = calculate_relative_velocity(vel1, vel2)
                approach_rate = calculate_approach_rate(pos1, pos2, vel1, vel2)
                
                results.append({
                    'timestamp': current_time,
                    'hours_from_start': step * interval_minutes / 60.0,
                    'distance_km': float(distance),
                    'relative_velocity_kmps': float(rel_velocity),
                    'approach_rate_kmps': float(approach_rate),
                    'position1': pos1.tolist(),
                    'position2': pos2.tolist(),
                    'velocity1': vel1.tolist(),
                    'velocity2': vel2.tolist()
                })
                
            except Exception as e:
                print(f"Warning: Propagation failed at {current_time}: {e}")
                continue
        
        return results
    
    def find_closest_approach(self, sat1_tle: Dict, sat2_tle: Dict,
                             start_time: datetime,
                             duration_hours: float = 24.0) -> Dict:
        """
        Find the closest approach between two satellites.
        
        Args:
            sat1_tle, sat2_tle: TLE data
            start_time: Search start time
            duration_hours: Search duration
            
        Returns:
            Dictionary with closest approach data
        """
        # Propagate pair
        trajectory = self.propagate_pair(
            sat1_tle, sat2_tle,
            start_time,
            duration_hours,
            interval_minutes=5  # Finer resolution for closest approach
        )
        
        if not trajectory:
            raise ValueError("Propagation failed")
        
        # Find minimum distance
        min_point = min(trajectory, key=lambda x: x['distance_km'])
        
        return {
            'time_of_closest_approach': min_point['timestamp'],
            'minimum_distance_km': min_point['distance_km'],
            'relative_velocity_kmps': min_point['relative_velocity_kmps'],
            'hours_from_now': min_point['hours_from_start']
        }
    
    def get_current_state(self, tle_data: Dict) -> Dict:
        """
        Get current orbital state of a satellite.
        
        Args:
            tle_data: TLE dictionary
            
        Returns:
            Current state vector and elements
        """
        return self.propagate_satellite(
            tle_data['tle_line1'],
            tle_data['tle_line2'],
            datetime.utcnow()
        )
