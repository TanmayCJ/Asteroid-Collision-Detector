"""
AstroGuard Backend API
======================
FastAPI backend for satellite collision prediction service.

Features:
- RESTful API for satellite tracking
- Real-time collision risk predictions
- Orbit propagation using SGP4
- ML model integration
- WebSocket support for live updates
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for ML imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.schemas import (
    SatelliteInfo,
    PredictionRequest,
    PredictionResponse,
    TimelineResponse,
    ScenarioRequest,
    ScenarioResponse,
    HealthResponse
)
from backend.services.orbit_service import OrbitPropagationService
from backend.services.ml_service import MLPredictionService
from backend.services.risk_service import RiskCalculationService
from backend.database import SatelliteDatabase

# Initialize FastAPI app
app = FastAPI(
    title="AstroGuard API",
    description="Satellite Collision Prediction Service using ML + SGP4",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
orbit_service = OrbitPropagationService()
ml_service = MLPredictionService()
risk_service = RiskCalculationService(ml_service, orbit_service)
db = SatelliteDatabase()


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information."""
    return {
        "service": "AstroGuard Collision Predictor",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "satellites": "/objects",
            "predict": "/predict",
            "timeline": "/timeline",
            "scenario": "/scenario"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns status of all services and components.
    """
    ml_status = ml_service.is_model_loaded()
    db_status = db.is_initialized()
    
    return HealthResponse(
        status="healthy" if ml_status and db_status else "degraded",
        timestamp=datetime.utcnow(),
        services={
            "ml_model": "operational" if ml_status else "unavailable",
            "database": "operational" if db_status else "unavailable",
            "orbit_propagator": "operational"
        }
    )


# ============================================================================
# SATELLITE DATA ENDPOINTS
# ============================================================================

@app.get("/objects", response_model=List[SatelliteInfo], tags=["Satellites"])
async def get_satellites(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of satellites to return"),
    orbit_regime: Optional[str] = Query(None, description="Filter by orbit regime: LEO, MEO, GEO")
):
    """
    Get list of tracked satellites with TLE and orbit information.
    
    Query Parameters:
    - limit: Maximum number of results (default: 100)
    - orbit_regime: Filter by orbit type (LEO, MEO, GEO)
    
    Returns:
    - List of satellite objects with TLE data
    """
    try:
        satellites = db.get_satellites(limit=limit, orbit_regime=orbit_regime)
        return satellites
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch satellites: {str(e)}")


@app.get("/objects/{satellite_id}", response_model=SatelliteInfo, tags=["Satellites"])
async def get_satellite(satellite_id: str):
    """
    Get detailed information for a specific satellite.
    
    Path Parameters:
    - satellite_id: NORAD ID or satellite name
    
    Returns:
    - Detailed satellite information
    """
    try:
        satellite = db.get_satellite_by_id(satellite_id)
        if not satellite:
            raise HTTPException(status_code=404, detail=f"Satellite {satellite_id} not found")
        return satellite
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COLLISION PREDICTION ENDPOINTS
# ============================================================================

@app.get("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_collision(
    objectA: str = Query(..., description="ID of first satellite"),
    objectB: str = Query(..., description="ID of second satellite"),
    start_time: Optional[str] = Query(None, description="Prediction start time (ISO format)")
):
    """
    Predict collision risk between two satellites using ML model.
    
    Query Parameters:
    - objectA: First satellite ID (NORAD ID or name)
    - objectB: Second satellite ID (NORAD ID or name)
    - start_time: Optional start time (default: now)
    
    Returns:
    - Predicted minimum distance in next 24 hours
    - Risk classification (SAFE, CAUTION, HIGH_RISK)
    - Current orbital parameters
    """
    try:
        # Get satellite data
        sat1 = db.get_satellite_by_id(objectA)
        sat2 = db.get_satellite_by_id(objectB)
        
        if not sat1 or not sat2:
            raise HTTPException(status_code=404, detail="One or both satellites not found")
        
        # Parse start time
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
        else:
            start_dt = datetime.utcnow()
        
        # Run prediction
        result = risk_service.predict_collision_risk(sat1, sat2, start_dt)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/timeline", response_model=TimelineResponse, tags=["Predictions"])
async def get_risk_timeline(
    objectA: str = Query(..., description="ID of first satellite"),
    objectB: str = Query(..., description="ID of second satellite"),
    hours: int = Query(24, ge=1, le=168, description="Prediction horizon in hours"),
    interval: int = Query(1, ge=1, le=24, description="Time interval between predictions (hours)")
):
    """
    Get risk predictions over time for a satellite pair.
    
    Generates a timeline showing how collision risk evolves.
    
    Query Parameters:
    - objectA, objectB: Satellite IDs
    - hours: Prediction horizon (default: 24h, max: 168h/7 days)
    - interval: Time between predictions (default: 1h)
    
    Returns:
    - Timeline of risk predictions
    - Minimum distance at each timestep
    - Risk level evolution
    """
    try:
        sat1 = db.get_satellite_by_id(objectA)
        sat2 = db.get_satellite_by_id(objectB)
        
        if not sat1 or not sat2:
            raise HTTPException(status_code=404, detail="One or both satellites not found")
        
        # Generate timeline
        timeline = risk_service.generate_risk_timeline(
            sat1, sat2,
            duration_hours=hours,
            interval_hours=interval
        )
        
        return TimelineResponse(
            satellite_a=sat1.name,
            satellite_b=sat2.name,
            timeline=timeline,
            prediction_start=datetime.utcnow(),
            duration_hours=hours
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline generation failed: {str(e)}")


# ============================================================================
# SCENARIO ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/scenario", response_model=ScenarioResponse, tags=["Scenarios"])
async def analyze_scenario(scenario: ScenarioRequest):
    """
    Analyze "what-if" scenarios with orbital maneuvers.
    
    Allows proposing orbital changes and recalculating collision risk.
    
    Request Body:
    - satellite_id: Which satellite to maneuver
    - delta_v: Velocity change vector (km/s)
    - maneuver_time: When to apply maneuver
    - target_pair: Satellite to check collision with
    
    Returns:
    - Risk before and after maneuver
    - Effectiveness of collision avoidance
    """
    try:
        # Get satellites
        sat_to_maneuver = db.get_satellite_by_id(scenario.satellite_id)
        target_sat = db.get_satellite_by_id(scenario.target_satellite_id)
        
        if not sat_to_maneuver or not target_sat:
            raise HTTPException(status_code=404, detail="Satellite not found")
        
        # Calculate scenario
        result = risk_service.analyze_maneuver_scenario(
            sat_to_maneuver,
            target_sat,
            scenario.delta_v,
            scenario.maneuver_time
        )
        
        return ScenarioResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")


# ============================================================================
# STATISTICS & ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/stats/high-risk", tags=["Statistics"])
async def get_high_risk_pairs(
    threshold: str = Query("CAUTION", description="Minimum risk level: SAFE, CAUTION, HIGH_RISK"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get list of high-risk satellite pairs.
    
    Returns pairs sorted by collision risk (most dangerous first).
    """
    try:
        high_risk = risk_service.get_high_risk_pairs(threshold=threshold, limit=limit)
        return high_risk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/orbit-density", tags=["Statistics"])
async def get_orbit_density():
    """
    Get statistics on orbital density by altitude.
    
    Returns distribution of satellites across orbit regimes.
    """
    try:
        density = db.get_orbit_density_stats()
        return density
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("\n" + "="*60)
    print("🚀 AstroGuard API Starting...")
    print("="*60)
    
    # Initialize database
    db.initialize()
    print("✓ Database initialized")
    
    # Load ML model
    ml_service.load_model()
    print("✓ ML model loaded")
    
    print("\n✓ API ready at http://localhost:8000")
    print("✓ Documentation at http://localhost:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n🛑 AstroGuard API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
