"""
Add sample satellites to the database for testing
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.database import satellites_db

# Sample TLE data for real satellites
sample_satellites = [
    {
        "id": "ISS",
        "name": "ISS (ZARYA)",
        "norad_id": "25544",
        "tle_line1": "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9005",
        "tle_line2": "2 25544  51.6400 247.4627 0002602  70.4458  32.0187 15.54225995 12345",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    },
    {
        "id": "HUBBLE",
        "name": "HST (HUBBLE SPACE TELESCOPE)",
        "norad_id": "20580",
        "tle_line1": "1 20580U 90037B   23001.00000000  .00000981  00000-0  52283-4 0  9991",
        "tle_line2": "2 20580  28.4699 288.8102 0002524 321.7771 171.5762 15.09686066 12345",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    },
    {
        "id": "STARLINK-1007",
        "name": "STARLINK-1007",
        "norad_id": "44713",
        "tle_line1": "1 44713U 19074A   23001.00000000  .00001234  00000-0  89012-4 0  9998",
        "tle_line2": "2 44713  53.0539  12.3456 0001234  90.1234 269.9876 15.06492345 67890",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    },
    {
        "id": "COSMOS-2251-DEB",
        "name": "COSMOS 2251 DEB",
        "norad_id": "34454",
        "tle_line1": "1 34454U 93036SVZ 23001.00000000  .00000456  00000-0  12345-3 0  9997",
        "tle_line2": "2 34454  74.0412 123.4567 0012345 234.5678  12.3456 14.34567890 12345",
        "object_type": "DEBRIS",
        "status": "DECAYED"
    },
    {
        "id": "FENGYUN-1C-DEB",
        "name": "FENGYUN 1C DEB",
        "norad_id": "30000",
        "tle_line1": "1 30000U 07006AAA 23001.00000000  .00000789  00000-0  45678-3 0  9996",
        "tle_line2": "2 30000  98.7654 345.6789 0023456 123.4567 236.7890 14.12345678 12345",
        "object_type": "DEBRIS",
        "status": "DECAYED"
    },
    {
        "id": "GOES-16",
        "name": "GOES-16",
        "norad_id": "41866",
        "tle_line1": "1 41866U 16071A   23001.00000000 -.00000123  00000-0  00000+0 0  9994",
        "tle_line2": "2 41866   0.0456  75.3456 0001234 123.4567 236.7890  1.00271234 12345",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    },
    {
        "id": "SENTINEL-1A",
        "name": "SENTINEL-1A",
        "norad_id": "39634",
        "tle_line1": "1 39634U 14016A   23001.00000000  .00000012  00000-0  12345-4 0  9993",
        "tle_line2": "2 39634  98.1823  12.3456 0001234  90.1234 269.9876 14.59198765 12345",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    },
    {
        "id": "TIANGONG",
        "name": "TIANGONG SPACE STATION",
        "norad_id": "48274",
        "tle_line1": "1 48274U 21035A   23001.00000000  .00012345  00000-0  98765-3 0  9992",
        "tle_line2": "2 48274  41.4678 123.4567 0003456 234.5678  12.3456 15.61234567 12345",
        "object_type": "PAYLOAD",
        "status": "OPERATIONAL"
    }
]

def add_satellites():
    """Add sample satellites to the database"""
    print("🛰️  Adding sample satellites to database...\n")
    
    for sat in sample_satellites:
        satellites_db[sat["id"]] = sat
        print(f"✓ Added: {sat['name']} ({sat['object_type']})")
    
    print(f"\n✅ Successfully added {len(sample_satellites)} satellites!")
    print(f"📊 Total satellites in database: {len(satellites_db)}")
    
    print("\n🎯 You can now:")
    print("1. Go to http://localhost:3000 - Frontend UI")
    print("2. Go to http://localhost:8000/docs - Try GET /objects endpoint")
    print("3. Test predictions with satellite pairs!")

if __name__ == "__main__":
    add_satellites()
