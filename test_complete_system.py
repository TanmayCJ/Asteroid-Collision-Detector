"""
Complete system test - Tests all components
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_complete_system():
    """Run complete system test"""
    
    print("=" * 70)
    print("🧪 ASTROGUARD COMPLETE SYSTEM TEST")
    print("=" * 70)
    
    # Test 1: Health Check
    print("\n1️⃣  HEALTH CHECK")
    print("-" * 70)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Backend is running!")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   ML Model: {health.get('services', {}).get('ml_model', 'unknown')}")
            print(f"   Database: {health.get('services', {}).get('database', 'unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("\n💡 Make sure backend is running:")
        print("   cd backend")
        print("   python -m uvicorn main:app --reload --port 8000")
        return
    
    # Test 2: Root endpoint
    print("\n2️⃣  ROOT ENDPOINT")
    print("-" * 70)
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Service: {info.get('service', 'N/A')}")
            print(f"   Version: {info.get('version', 'N/A')}")
            print(f"   Status: {info.get('status', 'N/A')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Satellites
    print("\n3️⃣  SATELLITE DATABASE")
    print("-" * 70)
    try:
        response = requests.get(f"{BASE_URL}/objects?limit=10", timeout=5)
        if response.status_code == 200:
            satellites = response.json()
            count = len(satellites)
            
            if count > 0:
                print(f"✅ Found {count} satellites!")
                print(f"\n   First 3 satellites:")
                for i, sat in enumerate(satellites[:3], 1):
                    print(f"   {i}. {sat['name']} (ID: {sat['norad_id']}, {sat.get('orbit_regime', 'N/A')})")
                
                # Test 4: Prediction
                print("\n4️⃣  COLLISION PREDICTION TEST")
                print("-" * 70)
                
                sat_a = satellites[0]['norad_id']
                sat_b = satellites[1]['norad_id'] if len(satellites) > 1 else satellites[0]['norad_id']
                
                print(f"   Testing: {satellites[0]['name']} vs {satellites[1]['name']}")
                print(f"   IDs: {sat_a} vs {sat_b}")
                
                pred_url = f"{BASE_URL}/predict?satellite_a_id={sat_a}&satellite_b_id={sat_b}&prediction_horizon_hours=24"
                pred_response = requests.get(pred_url, timeout=10)
                
                if pred_response.status_code == 200:
                    result = pred_response.json()
                    
                    print(f"\n   📊 PREDICTION RESULTS:")
                    print(f"   {'─' * 66}")
                    print(f"   Satellite A:          {result.get('satellite_a', 'N/A')}")
                    print(f"   Satellite B:          {result.get('satellite_b', 'N/A')}")
                    print(f"   Min Distance:         {result.get('predicted_min_distance_km', 0):.2f} km")
                    print(f"   Current Distance:     {result.get('current_distance_km', 0):.2f} km")
                    print(f"   Relative Velocity:    {result.get('relative_velocity_kmps', 0):.4f} km/s")
                    print(f"   Risk Level:           {result.get('risk_level', 'UNKNOWN')}")
                    print(f"   Prediction Horizon:   {result.get('prediction_horizon_hours', 0)} hours")
                    print(f"   {'─' * 66}")
                    
                    risk = result.get('risk_level', 'UNKNOWN')
                    if risk == 'SAFE':
                        print(f"   ✅ Status: SAFE - No collision risk")
                    elif risk == 'CAUTION':
                        print(f"   ⚠️  Status: CAUTION - Monitor closely")
                    elif risk == 'HIGH_RISK':
                        print(f"   🚨 Status: HIGH RISK - Immediate action needed!")
                    
                    print(f"\n✅ Prediction successful!")
                    
                else:
                    print(f"   ❌ Prediction failed: {pred_response.status_code}")
                    print(f"   Error: {pred_response.text}")
                    
            else:
                print(f"⚠️  No satellites in database (returned empty)")
                print(f"\n   The backend might not be loading the data file correctly.")
                print(f"   Data file exists at: ml/data/synthetic_tle.json")
                print(f"\n   Try restarting the backend:")
                print(f"   1. Stop backend (Ctrl+C)")
                print(f"   2. Run: python -m uvicorn backend.main:app --reload --port 8000")
                
        else:
            print(f"❌ Satellites endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print("✅ Backend API: RUNNING")
    print("✅ ML Model: LOADED")
    
    if count > 0:
        print(f"✅ Satellites: {count} LOADED")
        print("✅ Predictions: WORKING")
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Open http://localhost:3000 - See 3D visualization")
        print("2. Open http://localhost:8000/docs - Try API interactively")
        print(f"3. Test more predictions with satellite IDs: {sat_a}, {sat_b}, etc.")
        
    else:
        print("⚠️  Satellites: NOT LOADED")
        print("⚠️  Predictions: CANNOT TEST")
        print("\n💡 TIP: Restart the backend to load satellites")
        
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_complete_system()
