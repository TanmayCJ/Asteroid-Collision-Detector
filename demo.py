"""
🎯 QUICK DEMO - Test your collision predictor!
"""
import requests

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🛰️  ASTROGUARD COLLISION PREDICTOR - QUICK DEMO")
print("=" * 70)

# Get satellites
print("\n📡 Loading satellites...")
response = requests.get(f"{BASE_URL}/objects?limit=5")
satellites = response.json()

print(f"✅ Found {len(satellites)} satellites:\n")
for i, sat in enumerate(satellites, 1):
    print(f"   {i}. {sat['name']:12} (ID: {sat['norad_id']:6}) - {sat['orbit_regime']:3} orbit")

# Test prediction
if len(satellites) >= 2:
    sat_a = satellites[0]
    sat_b = satellites[1]
    
    print(f"\n🔮 Testing collision prediction...")
    print(f"   Satellite A: {sat_a['name']} (ID: {sat_a['norad_id']})")
    print(f"   Satellite B: {sat_b['name']} (ID: {sat_b['norad_id']})")
    
    # Make prediction
    pred_url = f"{BASE_URL}/predict?objectA={sat_a['norad_id']}&objectB={sat_b['norad_id']}"
    
    try:
        pred_response = requests.get(pred_url, timeout=15)
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            
            print(f"\n   ✨ PREDICTION RESULTS:")
            print(f"   ╔═{'═' * 64}╗")
            print(f"   ║ {'Predicted Min Distance:':30} {result.get('predicted_min_distance_km', 0):8.2f} km      ║")
            print(f"   ║ {'Current Distance:':30} {result.get('current_distance_km', 0):8.2f} km      ║")
            print(f"   ║ {'Relative Velocity:':30} {result.get('relative_velocity_kmps', 0):8.4f} km/s    ║")
            print(f"   ║ {'Prediction Horizon:':30} {result.get('prediction_horizon_hours', 0):8d} hours   ║")
            print(f"   ╠═{'═' * 64}╣")
            
            risk = result.get('risk_level', 'UNKNOWN')
            if risk == 'SAFE':
                emoji = "✅"
                status = "SAFE"
                msg = "No collision risk"
            elif risk == 'CAUTION':
                emoji = "⚠️ "
                status = "CAUTION"
                msg = "Monitor situation closely"
            elif risk == 'HIGH_RISK':
                emoji = "🚨"
                status = "HIGH RISK"
                msg = "Immediate action required!"
            else:
                emoji = "❓"
                status = "UNKNOWN"
                msg = "Unable to determine risk"
            
            print(f"   ║ {emoji} RISK LEVEL: {status:20}                      ║")
            print(f"   ║    {msg:50}       ║")
            print(f"   ╚═{'═' * 64}╝")
            
            print(f"\n✅ Prediction successful!")
            
        else:
            print(f"\n   ❌ Prediction failed: {pred_response.status_code}")
            print(f"   Error: {pred_response.text}")
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")

print("\n" + "=" * 70)
print("🎯 WHAT TO DO NEXT:")
print("=" * 70)
print()
print("1️⃣  INTERACTIVE API TESTING")
print("   Open: http://localhost:8000/docs")
print("   → Try different satellite pairs")
print("   → Experiment with parameters")
print()
print("2️⃣  3D VISUALIZATION")
print("   Open: http://localhost:3000")
print("   → See satellites orbiting Earth")
print("   → Interactive 3D controls")
print()
print("3️⃣  TEST MORE PREDICTIONS (PowerShell)")
print('   curl "http://localhost:8000/predict?objectA=10000&objectB=10002"')
print('   curl "http://localhost:8000/predict?objectA=10001&objectB=10003"')
print()
print("4️⃣  VIEW TIMELINE")
print('   curl "http://localhost:8000/timeline?objectA=10000&objectB=10001&hours=24"')
print()
print("=" * 70)
print("🎉 Your satellite collision prediction system is fully operational!")
print("=" * 70)
