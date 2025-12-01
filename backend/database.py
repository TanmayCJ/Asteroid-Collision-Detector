"""
In-memory satellite database.
In production, replace with PostgreSQL or similar.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.schemas import SatelliteInfo


class SatelliteDatabase:
    """
    Simple in-memory database for satellite TLE data.
    """
    
    def __init__(self, data_path: str = "ml/data/synthetic_tle.json"):
        """
        Initialize database.
        
        Args:
            data_path: Path to TLE data file
        """
        self.data_path = Path(data_path)
        self.satellites = []
        self._initialized = False
    
    def initialize(self):
        """Load satellite data from file."""
        if not self.data_path.exists():
            print(f"Warning: Satellite data not found at {self.data_path}")
            print("Generating sample data...")
            self._generate_sample_data()
        
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            self.satellites = data.get('satellites', [])
            self._initialized = True
            print(f"✓ Loaded {len(self.satellites)} satellites from database")
            
        except Exception as e:
            print(f"Error loading satellite data: {e}")
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if database is ready."""
        return self._initialized and len(self.satellites) > 0
    
    def get_satellites(self, limit: int = 100, 
                      orbit_regime: Optional[str] = None) -> List[SatelliteInfo]:
        """
        Get list of satellites.
        
        Args:
            limit: Maximum number to return
            orbit_regime: Filter by orbit type
            
        Returns:
            List of SatelliteInfo objects
        """
        filtered = self.satellites
        
        if orbit_regime:
            filtered = [s for s in filtered if s.get('orbit_regime') == orbit_regime]
        
        filtered = filtered[:limit]
        
        # Convert to SatelliteInfo objects
        result = []
        for sat in filtered:
            result.append(SatelliteInfo(
                name=sat['name'],
                norad_id=sat['norad_id'],
                tle_line1=sat['tle_line1'],
                tle_line2=sat['tle_line2'],
                altitude_km=sat.get('altitude_km', 0),
                orbit_regime=sat.get('orbit_regime', 'UNKNOWN'),
                last_updated=datetime.utcnow()
            ))
        
        return result
    
    def get_satellite_by_id(self, satellite_id: str) -> Optional[Dict]:
        """
        Get satellite by NORAD ID or name.
        
        Args:
            satellite_id: Satellite identifier
            
        Returns:
            Satellite dictionary or None
        """
        for sat in self.satellites:
            if sat['norad_id'] == satellite_id or sat['name'] == satellite_id:
                return sat
        
        return None
    
    def get_orbit_density_stats(self) -> Dict:
        """
        Get statistics on orbit density.
        
        Returns:
            Dictionary with orbit regime statistics
        """
        if not self.is_initialized():
            return {}
        
        regimes = {}
        for sat in self.satellites:
            regime = sat.get('orbit_regime', 'UNKNOWN')
            regimes[regime] = regimes.get(regime, 0) + 1
        
        total = len(self.satellites)
        
        return {
            'total_satellites': total,
            'by_regime': regimes,
            'percentages': {
                regime: (count / total * 100) for regime, count in regimes.items()
            }
        }
    
    def _generate_sample_data(self):
        """Generate sample satellite data if none exists."""
        print("Generating sample satellite data...")
        
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from ml.preprocessing import OrbitDataPreprocessor
            
            preprocessor = OrbitDataPreprocessor()
            preprocessor.generate_synthetic_dataset(
                num_satellites=50,
                output_path=str(self.data_path)
            )
            
        except Exception as e:
            print(f"Failed to generate sample data: {e}")
