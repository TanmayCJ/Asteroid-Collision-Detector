"""
AstroGuard ML Utilities
========================
Core utility functions for orbit calculations, distance metrics, and data transformations.

Author: AstroGuard Team
"""

import numpy as np
from sgp4.api import jday
from sgp4.api import Satrec
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import math
import json


def tle_to_cartesian(tle_line1: str, tle_line2: str, time: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert TLE (Two-Line Element) to Cartesian coordinates using SGP4.
    
    Args:
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
        time: Datetime object for propagation
        
    Returns:
        position: (x, y, z) in km
        velocity: (vx, vy, vz) in km/s
    """
    try:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        
        # Check satellite initialization
        if satellite.error != 0:
            print(f"[DEBUG] Satellite init error: {satellite.error}")
            raise ValueError(f"Satellite initialization error: {satellite.error}")
        
        jd, fr = jday(time.year, time.month, time.day, 
                      time.hour, time.minute, time.second)
        
        error_code, position, velocity = satellite.sgp4(jd, fr)
        
        if error_code != 0:
            print(f"[DEBUG] SGP4 propagation error: {error_code}")
            raise ValueError(f"SGP4 error code: {error_code}")
            
        return np.array(position), np.array(velocity)
    except Exception as e:
        print(f"[DEBUG] TLE propagation exception: {str(e)}")
        raise ValueError(f"TLE propagation failed: {str(e)}")


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two position vectors.
    
    Args:
        pos1: Position vector 1 (x, y, z) in km
        pos2: Position vector 2 (x, y, z) in km
        
    Returns:
        Distance in km
    """
    return np.linalg.norm(pos1 - pos2)


def calculate_relative_velocity(vel1: np.ndarray, vel2: np.ndarray) -> float:
    """
    Calculate magnitude of relative velocity between two objects.
    
    Args:
        vel1: Velocity vector 1 (vx, vy, vz) in km/s
        vel2: Velocity vector 2 (vx, vy, vz) in km/s
        
    Returns:
        Relative velocity magnitude in km/s
    """
    relative_vel = vel1 - vel2
    return np.linalg.norm(relative_vel)


def calculate_approach_rate(pos1: np.ndarray, pos2: np.ndarray, 
                            vel1: np.ndarray, vel2: np.ndarray) -> float:
    """
    Calculate the rate at which two objects are approaching (or separating).
    Negative value = approaching, Positive = separating.
    
    Args:
        pos1, pos2: Position vectors in km
        vel1, vel2: Velocity vectors in km/s
        
    Returns:
        Approach rate in km/s (negative = closing)
    """
    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1
    distance = np.linalg.norm(relative_pos)
    
    if distance == 0:
        return 0.0
        
    # Rate of change of distance
    approach_rate = np.dot(relative_pos, relative_vel) / distance
    return approach_rate


def calculate_relative_features(pos1: np.ndarray, pos2: np.ndarray,
                                vel1: np.ndarray, vel2: np.ndarray) -> Dict[str, float]:
    """
    Calculate all relative features between two satellites.
    
    Args:
        pos1, pos2: Position vectors in km
        vel1, vel2: Velocity vectors in km/s
        
    Returns:
        Dictionary with relative features
    """
    distance = calculate_distance(pos1, pos2)
    rel_velocity = calculate_relative_velocity(vel1, vel2)
    approach_rate = calculate_approach_rate(pos1, pos2, vel1, vel2)
    
    return {
        'distance': distance,
        'relative_velocity': rel_velocity,
        'approach_rate': approach_rate
    }


def orbital_elements(position: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
    """
    Calculate classical orbital elements from state vectors.
    
    Args:
        position: Position vector (x, y, z) in km
        velocity: Velocity vector (vx, vy, vz) in km/s
        
    Returns:
        Dictionary with orbital elements
    """
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    
    # Specific angular momentum
    h_vec = np.cross(position, velocity)
    h = np.linalg.norm(h_vec)
    
    # Eccentricity vector
    e_vec = (np.cross(velocity, h_vec) / mu) - (position / r)
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    energy = (v**2 / 2) - (mu / r)
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')
    
    # Inclination
    i = math.acos(h_vec[2] / h) * 180 / math.pi
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': i,
        'specific_angular_momentum': h
    }


def time_to_closest_approach(pos1: np.ndarray, pos2: np.ndarray,
                             vel1: np.ndarray, vel2: np.ndarray) -> float:
    """
    Estimate time to closest approach assuming linear motion.
    
    Args:
        pos1, pos2: Position vectors in km
        vel1, vel2: Velocity vectors in km/s
        
    Returns:
        Time to closest approach in seconds (0 if already at minimum)
    """
    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1
    
    # Using calculus: minimize |r(t)|^2 where r(t) = relative_pos + relative_vel * t
    # Derivative = 0 when: t = -dot(relative_pos, relative_vel) / dot(relative_vel, relative_vel)
    
    vel_squared = np.dot(relative_vel, relative_vel)
    
    if vel_squared < 1e-10:
        return 0.0
    
    t = -np.dot(relative_pos, relative_vel) / vel_squared
    
    return max(0.0, t)


def propagate_orbit_linear(position: np.ndarray, velocity: np.ndarray, 
                           dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple linear propagation (for short time periods only).
    For production use, prefer SGP4 propagation.
    
    Args:
        position: Initial position (x, y, z) in km
        velocity: Initial velocity (vx, vy, vz) in km/s
        dt_seconds: Time step in seconds
        
    Returns:
        new_position, new_velocity
    """
    new_position = position + velocity * dt_seconds
    return new_position, velocity


def normalize_features(features: np.ndarray, mean: np.ndarray = None, 
                      std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array (samples, features)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)
        
    Returns:
        normalized_features, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std


def risk_classification(min_distance_km: float, relative_velocity_kmps: float) -> str:
    """
    Classify collision risk based on minimum distance and relative velocity.
    
    Risk Thresholds:
    - HIGH: < 5 km with high velocity (> 1 km/s)
    - CAUTION: 5-25 km or moderate velocity
    - SAFE: > 25 km
    
    Args:
        min_distance_km: Predicted minimum distance in km
        relative_velocity_kmps: Relative velocity in km/s
        
    Returns:
        Risk level: 'SAFE', 'CAUTION', or 'HIGH_RISK'
    """
    if min_distance_km < 5.0 and relative_velocity_kmps > 1.0:
        return 'HIGH_RISK'
    elif min_distance_km < 5.0 or (min_distance_km < 25.0 and relative_velocity_kmps > 2.0):
        return 'CAUTION'
    elif min_distance_km < 25.0:
        return 'CAUTION'
    else:
        return 'SAFE'


def generate_synthetic_tle(semi_major_axis_km: float, eccentricity: float, 
                           inclination_deg: float, object_name: str = "SAT") -> Tuple[str, str]:
    """
    Generate synthetic TLE for testing purposes.
    Note: This creates approximate TLE format - not for operational use.
    
    Args:
        semi_major_axis_km: Semi-major axis in km
        eccentricity: Orbital eccentricity (0-1)
        inclination_deg: Inclination in degrees
        object_name: Satellite name
        
    Returns:
        tle_line1, tle_line2
    """
    # Mean motion (revs/day) from semi-major axis
    mu = 398600.4418  # km^3/s^2
    period_sec = 2 * math.pi * math.sqrt(semi_major_axis_km**3 / mu)
    mean_motion = 86400 / period_sec  # revolutions per day
    
    # Create properly formatted TLE lines with checksums
    # Line 1 format
    line1_base = f"1 99999U 24001A   24335.50000000  .00000000  00000-0  00000+0 0  999"
    checksum1 = _tle_checksum(line1_base)
    line1 = line1_base + str(checksum1)
    
    # Line 2 format - ensure proper field widths
    ecc_str = f"{int(eccentricity*10000000):07d}"
    line2_base = f"2 99999 {inclination_deg:>8.4f}   0.0000 {ecc_str}   0.0000   0.0000 {mean_motion:>11.8f}    0"
    checksum2 = _tle_checksum(line2_base)
    line2 = line2_base + str(checksum2)
    
    return line1, line2


def _tle_checksum(line: str) -> int:
    """Calculate TLE checksum."""
    checksum = 0
    for char in line:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10


def collision_probability(miss_distance_km: float, combined_radius_km: float = 0.01,
                          position_uncertainty_km: float = 5.0) -> float:
    """
    Calculate approximate collision probability using a simple model.
    
    Args:
        miss_distance_km: Predicted miss distance
        combined_radius_km: Sum of object radii (default 10m for satellites)
        position_uncertainty_km: Position uncertainty (default 5km for realistic tracking)
        
    Returns:
        Probability (0-1)
    """
    # For very close approaches, use distance-based probability
    if miss_distance_km < 1.0:
        return 0.95
    elif miss_distance_km < 5.0:
        # Linear interpolation between 95% at 1km and 60% at 5km
        return 0.95 - ((miss_distance_km - 1.0) / 4.0) * 0.35
    elif miss_distance_km < 10.0:
        # Linear interpolation between 60% at 5km and 30% at 10km
        return 0.60 - ((miss_distance_km - 5.0) / 5.0) * 0.30
    elif miss_distance_km < 25.0:
        # Linear interpolation between 30% at 10km and 10% at 25km
        return 0.30 - ((miss_distance_km - 10.0) / 15.0) * 0.20
    
    # For larger distances, use exponential decay
    sigma = position_uncertainty_km
    collision_threshold = combined_radius_km
    prob = math.exp(-0.5 * ((miss_distance_km - collision_threshold) / sigma) ** 2)
    prob = max(0.0, min(0.10, prob))
    
    return prob


def load_tle_data(filepath: str) -> list:
    """
    Load TLE data from JSON file.
    
    Args:
        filepath: Path to JSON file with TLE data
        
    Returns:
        List of satellite dictionaries
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Constants for orbit calculations
EARTH_RADIUS_KM = 6371.0
GEO_ALTITUDE_KM = 35786.0
LEO_ALTITUDE_MAX_KM = 2000.0
MEO_ALTITUDE_MAX_KM = 35786.0


def classify_orbit_regime(altitude_km: float) -> str:
    """
    Classify orbit by altitude regime.
    
    Args:
        altitude_km: Altitude above Earth surface
        
    Returns:
        Orbit regime: 'LEO', 'MEO', 'GEO', or 'HEO'
    """
    if altitude_km < LEO_ALTITUDE_MAX_KM:
        return 'LEO'
    elif altitude_km < MEO_ALTITUDE_MAX_KM:
        return 'MEO'
    elif abs(altitude_km - GEO_ALTITUDE_KM) < 500:
        return 'GEO'
    else:
        return 'HEO'
