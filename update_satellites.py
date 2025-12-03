"""
Update satellite JSON data with proper NORAD IDs and altitude calculations.
"""

import json
from sgp4.api import Satrec, jday
from datetime import datetime

def calculate_altitude_from_tle(tle_line1, tle_line2):
    """Calculate altitude from TLE."""
    try:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        now = datetime.utcnow()
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
        error_code, position, velocity = satellite.sgp4(jd, fr)
        
        if error_code == 0:
            # Calculate altitude from position
            import math
            r = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
            altitude_km = r - 6371.0  # Earth radius
            return max(0, altitude_km)
    except:
        pass
    return 500.0  # Default fallback

def classify_orbit(altitude_km):
    """Classify orbit regime."""
    if altitude_km < 2000:
        return 'LEO'
    elif altitude_km < 35786:
        return 'MEO'
    else:
        return 'GEO'

def main():
    # Read existing data
    with open('ml/ml/data/synthetic_tle.json', 'r') as f:
        satellites = json.load(f)
    
    # Update each satellite
    updated_satellites = []
    norad_base = 10000
    
    for idx, sat in enumerate(satellites):
        # Calculate altitude
        altitude = calculate_altitude_from_tle(sat['tle_line1'], sat['tle_line2'])
        orbit_regime = classify_orbit(altitude)
        
        # Add/update fields
        sat['norad_id'] = str(norad_base + idx)
        sat['altitude_km'] = altitude
        sat['orbit_regime'] = orbit_regime
        
        updated_satellites.append(sat)
        print(f"✓ {sat['name']}: {altitude:.1f} km ({orbit_regime})")
    
    # Save updated data
    output = {"satellites": updated_satellites}
    with open('ml/ml/data/synthetic_tle.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Updated {len(updated_satellites)} satellites")
    print(f"✓ Saved to ml/ml/data/synthetic_tle.json")

if __name__ == "__main__":
    main()
