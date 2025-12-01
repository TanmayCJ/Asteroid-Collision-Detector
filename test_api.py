"""
Quick API Test Script
=====================
Tests the AstroGuard backend API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint."""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_satellites():
    """Test satellites listing."""
    print("\n" + "="*60)
    print("Testing Satellites Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/objects?limit=5")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} satellites")
        if len(data) > 0:
            print(f"First satellite: {data[0].get('name', 'Unknown')}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ASTROGUARD API TEST SUITE")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Satellites List", test_satellites),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
