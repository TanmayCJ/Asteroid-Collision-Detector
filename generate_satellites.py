"""
Generate synthetic satellite data for the database
"""
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from ml.utils import generate_synthetic_tle

def generate_satellite_data(num_satellites=50):
    """Generate synthetic satellites with realistic orbital parameters"""
    
    print(f"🛰️  Generating {num_satellites} synthetic satellites...")
    
    satellites = []
    np.random.seed(42)
    
    orbit_regimes = {
        'LEO': (6600, 8000),      # Low Earth Orbit
        'MEO': (8000, 35786),     # Medium Earth Orbit  
        'GEO': (35786, 42164),    # Geosynchronous
    }
    
    for i in range(num_satellites):
        # Pick random orbit regime
        regime = np.random.choice(['LEO', 'MEO', 'GEO'], p=[0.7, 0.2, 0.1])
        altitude_range = orbit_regimes[regime]
        
        # Generate realistic orbital parameters
        semi_major_axis = np.random.uniform(*altitude_range)
        eccentricity = np.random.uniform(0.0001, 0.02)  # Nearly circular
        inclination = np.random.uniform(0, 98)
        
        # Generate TLE
        tle_line1, tle_line2 = generate_synthetic_tle(
            semi_major_axis, eccentricity, inclination, f"SAT-{i:04d}"
        )
        
        # Create satellite entry
        satellites.append({
            'name': f"SAT-{i:04d}",
            'norad_id': f"{10000 + i}",
            'tle_line1': tle_line1,
            'tle_line2': tle_line2,
            'altitude_km': semi_major_axis - 6371,  # Earth radius
            'orbit_regime': regime,
            'semi_major_axis': semi_major_axis,
            'eccentricity': eccentricity,
            'inclination': inclination
        })
    
    # Save to file
    output_path = Path('ml/data/synthetic_tle.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'satellites': satellites,
        'generated_at': '2025-12-02T00:00:00Z',
        'count': len(satellites)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Generated {len(satellites)} satellites")
    print(f"✓ Saved to: {output_path}")
    
    # Print statistics
    regimes = {}
    for sat in satellites:
        regime = sat['orbit_regime']
        regimes[regime] = regimes.get(regime, 0) + 1
    
    print(f"\n📊 Orbit Distribution:")
    for regime, count in regimes.items():
        print(f"   {regime}: {count} satellites ({count/len(satellites)*100:.1f}%)")
    
    return satellites

if __name__ == "__main__":
    satellites = generate_satellite_data(50)
    
    print(f"\n✅ Database ready!")
    print(f"🎯 Restart the backend to load satellites:")
    print(f"   cd backend && python -m uvicorn main:app --reload")
    print(f"\n   Then test at: http://localhost:8000/objects")
