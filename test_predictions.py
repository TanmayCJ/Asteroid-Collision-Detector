"""
Test the prediction endpoint with real satellite pairs
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_predictions():
    """Test collision predictions with different satellite pairs"""
    
    print("🧪 Testing AstroGuard Collision Predictions\n")
    print("=" * 60)
    
    # First, check if we have satellites
    print("\n1️⃣  Checking available satellites...")
    response = requests.get(f"{BASE_URL}/objects?limit=10")
    
    if response.status_code == 200:
        satellites = response.json()
        print(f"   ✓ Found {len(satellites)} satellites in database")
        
        if len(satellites) >= 2:
            # Show first few satellites
            print("\n   Available satellites:")
            for i, sat in enumerate(satellites[:5], 1):
                print(f"   {i}. {sat['name']} (NORAD: {sat['norad_id']})")
            
            # Test prediction with first two satellites
            print(f"\n2️⃣  Testing prediction: {satellites[0]['name']} vs {satellites[1]['name']}")
            
            pred_response = requests.get(
                f"{BASE_URL}/predict",
                params={
                    "satellite_a_id": satellites[0]['norad_id'],
                    "satellite_b_id": satellites[1]['norad_id'],
                    "prediction_horizon_hours": 24
                }
            )
            
            if pred_response.status_code == 200:
                result = pred_response.json()
                print("\n   📊 PREDICTION RESULTS:")
                print(f"   {'─' * 50}")
                print(f"   Satellite A: {result.get('satellite_a', 'N/A')}")
                print(f"   Satellite B: {result.get('satellite_b', 'N/A')}")
                print(f"   Predicted Min Distance: {result.get('predicted_min_distance_km', 0):.2f} km")
                print(f"   Current Distance: {result.get('current_distance_km', 0):.2f} km")
                print(f"   Relative Velocity: {result.get('relative_velocity_kmps', 0):.4f} km/s")
                print(f"   Risk Level: {result.get('risk_level', 'UNKNOWN')}")
                print(f"   Prediction Horizon: {result.get('prediction_horizon_hours', 0)} hours")
                print(f"   {'─' * 50}")
                
                # Color code the risk
                risk = result.get('risk_level', 'UNKNOWN')
                if risk == 'SAFE':
                    print(f"   ✅ Status: SAFE - No collision risk")
                elif risk == 'CAUTION':
                    print(f"   ⚠️  Status: CAUTION - Monitor closely")
                elif risk == 'HIGH_RISK':
                    print(f"   🚨 Status: HIGH RISK - Immediate action needed!")
                
            else:
                print(f"   ❌ Prediction failed: {pred_response.status_code}")
                print(f"   Error: {pred_response.text}")
        else:
            print("   ⚠️  Not enough satellites for prediction (need at least 2)")
            print("   Run the backend to auto-generate sample data")
    else:
        print(f"   ❌ Failed to get satellites: {response.status_code}")
    
    # Test timeline endpoint
    print(f"\n3️⃣  Testing timeline analysis...")
    timeline_response = requests.get(f"{BASE_URL}/timeline")
    
    if timeline_response.status_code == 200:
        timeline = timeline_response.json()
        print(f"   ✓ Got timeline with {len(timeline.get('timeline', []))} points")
    else:
        print(f"   ❌ Timeline failed: {timeline_response.status_code}")
    
    # Test stats
    print(f"\n4️⃣  Getting system statistics...")
    stats_response = requests.get(f"{BASE_URL}/stats")
    
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"   ✓ Total satellites: {stats.get('total_satellites', 0)}")
        print(f"   ✓ Active predictions: {stats.get('active_predictions', 0)}")
        print(f"   ✓ High risk pairs: {stats.get('high_risk_pairs', 0)}")
    else:
        print(f"   ❌ Stats failed: {stats_response.status_code}")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!\n")
    
    print("🎯 Next steps:")
    print("1. Open http://localhost:3000 - See predictions in 3D")
    print("2. Open http://localhost:8000/docs - Try more endpoints")
    print("3. Experiment with different satellite pairs")

if __name__ == "__main__":
    try:
        test_predictions()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to backend at http://localhost:8000")
        print("Make sure the backend is running:")
        print("   cd backend && python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
