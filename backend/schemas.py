"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class SatelliteInfo(BaseModel):
    """Satellite information response."""
    name: str = Field(..., description="Satellite name")
    norad_id: str = Field(..., description="NORAD catalog ID")
    tle_line1: str = Field(..., description="TLE line 1")
    tle_line2: str = Field(..., description="TLE line 2")
    altitude_km: float = Field(..., description="Average altitude in km")
    orbit_regime: str = Field(..., description="Orbit type: LEO, MEO, GEO")
    last_updated: datetime = Field(..., description="TLE epoch")


class PredictionRequest(BaseModel):
    """Request for collision prediction."""
    satellite_a_id: str = Field(..., description="First satellite ID")
    satellite_b_id: str = Field(..., description="Second satellite ID")
    start_time: Optional[datetime] = Field(None, description="Prediction start time")


class PredictionResponse(BaseModel):
    """Collision prediction response."""
    satellite_a: str = Field(..., description="First satellite name")
    satellite_b: str = Field(..., description="Second satellite name")
    predicted_min_distance_km: float = Field(..., description="Predicted minimum distance")
    current_distance_km: float = Field(..., description="Current distance")
    relative_velocity_kmps: float = Field(..., description="Relative velocity magnitude")
    risk_level: str = Field(..., description="Risk classification: SAFE, CAUTION, HIGH_RISK")
    prediction_horizon_hours: int = Field(..., description="Prediction time horizon")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    confidence: Optional[float] = Field(None, description="Model confidence (0-1)")


class TimelinePoint(BaseModel):
    """Single point in risk timeline."""
    hours_from_now: int = Field(..., description="Hours from prediction start")
    distance_km: float = Field(..., description="Predicted distance at this time")
    risk_level: str = Field(..., description="Risk level at this time")
    timestamp: datetime = Field(..., description="Absolute timestamp")


class TimelineResponse(BaseModel):
    """Timeline of risk predictions."""
    satellite_a: str
    satellite_b: str
    timeline: List[TimelinePoint]
    prediction_start: datetime
    duration_hours: int


class ScenarioRequest(BaseModel):
    """Request for scenario analysis."""
    satellite_id: str = Field(..., description="Satellite to maneuver")
    target_satellite_id: str = Field(..., description="Satellite to avoid")
    delta_v: List[float] = Field(..., description="Velocity change [vx, vy, vz] in km/s")
    maneuver_time: datetime = Field(..., description="When to apply maneuver")


class ScenarioResponse(BaseModel):
    """Scenario analysis response."""
    satellite: str
    target: str
    risk_before_maneuver: str
    risk_after_maneuver: str
    min_distance_before_km: float
    min_distance_after_km: float
    maneuver_effectiveness: str = Field(..., description="Effectiveness rating")
    delta_v_magnitude_mps: float = Field(..., description="Maneuver delta-v in m/s")


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="Overall status: healthy, degraded, down")
    timestamp: datetime
    services: Dict[str, str] = Field(..., description="Status of individual services")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="Error timestamp")
