'use client'

import { useRef, useEffect, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, Sphere, Line } from '@react-three/drei'
import * as THREE from 'three'

function Earth() {
  return (
    <Sphere args={[1, 64, 64]} position={[0, 0, 0]}>
      <meshStandardMaterial
        color="#1e40af"
        emissive="#1e3a8a"
        emissiveIntensity={0.2}
        roughness={0.8}
      />
    </Sphere>
  )
}

function Satellite({ satellite, color, label }: any) {
  const meshRef = useRef<THREE.Mesh>(null)
  const groupRef = useRef<THREE.Group>(null)
  
  // Calculate orbital position based on satellite altitude
  const earthRadius = 1 // Three.js units
  const altitudeKm = satellite?.altitude_km || 500
  const orbitRadius = earthRadius + (altitudeKm / 6371) * 2 // Scale altitude to visible range
  
  // Create unique orbital position based on satellite NORAD ID
  const noradId = parseInt(satellite?.norad_id || '10000')
  const inclination = ((noradId * 137.5) % 180) * (Math.PI / 180) // Golden angle for distribution
  const raan = ((noradId * 222.5) % 360) * (Math.PI / 180) // Right ascension
  const argPerigee = ((noradId * 90) % 360) * (Math.PI / 180)
  
  // Animate satellite along orbit
  useFrame((state) => {
    if (groupRef.current) {
      const time = state.clock.getElapsedTime() * 0.1
      const angle = time + (noradId % 100) / 100 * Math.PI * 2
      
      // Calculate position in orbital plane
      const x = orbitRadius * Math.cos(angle)
      const z = orbitRadius * Math.sin(angle)
      const y = 0
      
      // Apply orbital inclination and rotation
      groupRef.current.position.x = x * Math.cos(raan) - z * Math.sin(raan)
      groupRef.current.position.y = y * Math.sin(inclination) + (x * Math.sin(raan) + z * Math.cos(raan)) * Math.cos(inclination)
      groupRef.current.position.z = y * Math.cos(inclination) - (x * Math.sin(raan) + z * Math.cos(raan)) * Math.sin(inclination)
    }
    
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.02
    }
  })

  return (
    <group ref={groupRef}>
      <Sphere ref={meshRef} args={[0.08, 16, 16]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.9}
        />
      </Sphere>
      {/* Orbit ring */}
      <mesh rotation={[Math.PI / 2 - inclination, 0, raan]}>
        <torusGeometry args={[orbitRadius, 0.008, 16, 100]} />
        <meshBasicMaterial color={color} transparent opacity={0.4} />
      </mesh>
      {/* Satellite label */}
      <sprite position={[0, 0.15, 0]} scale={[0.5, 0.2, 1]}>
        <spriteMaterial color={color} opacity={0.8} />
      </sprite>
    </group>
  )
}

function OrbitPath({ satellite }: any) {
  const points = []
  const radius = 2 + Math.random() * 1.5
  
  for (let i = 0; i <= 64; i++) {
    const angle = (i / 64) * Math.PI * 2
    points.push(new THREE.Vector3(
      Math.cos(angle) * radius,
      Math.sin(angle) * radius * 0.3,
      Math.sin(angle) * radius
    ))
  }
  
  return (
    <Line
      points={points}
      color="#00d9ff"
      lineWidth={1}
      transparent
      opacity={0.4}
    />
  )
}

function CollisionWarning({ prediction }: any) {
  const ringRef = useRef<THREE.Mesh>(null)
  
  if (!prediction || prediction.risk_level === 'SAFE') return null
  
  const color = prediction.risk_level === 'HIGH_RISK' ? '#ef4444' : '#f59e0b'
  
  useFrame((state) => {
    if (ringRef.current) {
      // Pulsing effect
      const pulse = Math.sin(state.clock.getElapsedTime() * 3) * 0.2 + 1
      ringRef.current.scale.setScalar(pulse)
      ringRef.current.rotation.z += 0.02
    }
  })
  
  return (
    <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[3, 0.05, 16, 100]} />
      <meshBasicMaterial color={color} transparent opacity={0.6} />
    </mesh>
  )
}

function TrajectoryPath({ satellite, color, steps = 100 }: any) {
  console.log('🛰️ TrajectoryPath rendering for satellite:', satellite?.name, 'color:', color)
  
  const curve = useMemo(() => {
    const pts = []
    const earthRadius = 1
    const altitudeKm = satellite?.altitude_km || 500
    const orbitRadius = earthRadius + (altitudeKm / 6371) * 2
    
    const noradId = parseInt(satellite?.norad_id || '10000')
    const inclination = ((noradId * 137.5) % 180) * (Math.PI / 180)
    const raan = ((noradId * 222.5) % 360) * (Math.PI / 180)
    
    // Generate future positions along orbit
    for (let i = 0; i <= steps; i++) {
      const angle = (i / steps) * Math.PI * 2
      const x = orbitRadius * Math.cos(angle)
      const z = orbitRadius * Math.sin(angle)
      const y = 0
      
      // Apply orbital transformations
      const posX = x * Math.cos(raan) - z * Math.sin(raan)
      const posY = y * Math.sin(inclination) + (x * Math.sin(raan) + z * Math.cos(raan)) * Math.cos(inclination)
      const posZ = y * Math.cos(inclination) - (x * Math.sin(raan) + z * Math.cos(raan)) * Math.sin(inclination)
      
      pts.push(new THREE.Vector3(posX, posY, posZ))
    }
    
    return new THREE.CatmullRomCurve3(pts, true)
  }, [satellite, steps])
  
  const tubeGeometry = useMemo(() => {
    return new THREE.TubeGeometry(curve, 100, 0.02, 8, true)
  }, [curve])
  
  return (
    <mesh geometry={tubeGeometry}>
      <meshBasicMaterial 
        color={color} 
        transparent 
        opacity={0.8}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function CollisionPoint({ satellites, prediction }: any) {
  const [collisionPos, setCollisionPos] = useState<THREE.Vector3 | null>(null)
  const markerRef = useRef<THREE.Mesh>(null)
  
  useEffect(() => {
    console.log('💥 CollisionPoint effect - prediction:', prediction)
    console.log('💥 CollisionPoint effect - satellites:', satellites)
    if (!prediction || prediction.risk_level === 'SAFE' || !satellites || satellites.length < 2) {
      console.log('💥 CollisionPoint: Not showing (safe or missing data)')
      setCollisionPos(null)
      return
    }
    
    console.log('💥 CollisionPoint: CALCULATING collision point')
    // Find approximate collision point by checking when satellites are closest
    const earthRadius = 1
    const sat1 = satellites[0]
    const sat2 = satellites[1]
    
    if (!sat1 || !sat2) return
    
    const orbitRadius1 = earthRadius + ((sat1.altitude_km || 500) / 6371) * 2
    const orbitRadius2 = earthRadius + ((sat2.altitude_km || 500) / 6371) * 2
    
    const noradId1 = parseInt(sat1.norad_id || '10000')
    const noradId2 = parseInt(sat2.norad_id || '10000')
    
    const inclination1 = ((noradId1 * 137.5) % 180) * (Math.PI / 180)
    const raan1 = ((noradId1 * 222.5) % 360) * (Math.PI / 180)
    const inclination2 = ((noradId2 * 137.5) % 180) * (Math.PI / 180)
    const raan2 = ((noradId2 * 222.5) % 360) * (Math.PI / 180)
    
    let minDist = Infinity
    let closestPos = new THREE.Vector3()
    
    // Check 200 points along the orbit to find closest approach
    for (let i = 0; i <= 200; i++) {
      const angle1 = (i / 200) * Math.PI * 2
      const angle2 = angle1 + (noradId1 % 100) / 100 * Math.PI * 2 - (noradId2 % 100) / 100 * Math.PI * 2
      
      // Position 1
      const x1 = orbitRadius1 * Math.cos(angle1)
      const z1 = orbitRadius1 * Math.sin(angle1)
      const pos1X = x1 * Math.cos(raan1) - z1 * Math.sin(raan1)
      const pos1Y = (x1 * Math.sin(raan1) + z1 * Math.cos(raan1)) * Math.cos(inclination1)
      const pos1Z = -(x1 * Math.sin(raan1) + z1 * Math.cos(raan1)) * Math.sin(inclination1)
      
      // Position 2
      const x2 = orbitRadius2 * Math.cos(angle2)
      const z2 = orbitRadius2 * Math.sin(angle2)
      const pos2X = x2 * Math.cos(raan2) - z2 * Math.sin(raan2)
      const pos2Y = (x2 * Math.sin(raan2) + z2 * Math.cos(raan2)) * Math.cos(inclination2)
      const pos2Z = -(x2 * Math.sin(raan2) + z2 * Math.cos(raan2)) * Math.sin(inclination2)
      
      const dist = Math.sqrt(
        Math.pow(pos1X - pos2X, 2) +
        Math.pow(pos1Y - pos2Y, 2) +
        Math.pow(pos1Z - pos2Z, 2)
      )
      
      if (dist < minDist) {
        minDist = dist
        closestPos = new THREE.Vector3(
          (pos1X + pos2X) / 2,
          (pos1Y + pos2Y) / 2,
          (pos1Z + pos2Z) / 2
        )
      }
    }
    
    console.log('💥 CollisionPoint calculated at:', closestPos, 'minDist:', minDist)
    setCollisionPos(closestPos)
  }, [satellites, prediction])
  
  useFrame((state) => {
    if (markerRef.current) {
      // Pulsing and rotating collision marker
      const pulse = Math.sin(state.clock.getElapsedTime() * 4) * 0.15 + 1
      markerRef.current.scale.setScalar(pulse)
      markerRef.current.rotation.x += 0.03
      markerRef.current.rotation.y += 0.02
    }
  })
  
  if (!collisionPos || prediction?.risk_level === 'SAFE') return null
  
  const color = prediction?.risk_level === 'HIGH_RISK' ? '#ef4444' : '#f59e0b'
  
  return (
    <group position={collisionPos}>
      {/* Main collision marker - octahedron */}
      <mesh ref={markerRef}>
        <octahedronGeometry args={[0.15, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={2}
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Warning rings around collision point */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.3, 0.02, 16, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
      <mesh rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[0.3, 0.02, 16, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[0.3, 0.02, 16, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
      
      {/* Outer glow sphere */}
      <Sphere args={[0.4, 16, 16]}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.2}
          side={THREE.BackSide}
        />
      </Sphere>
    </group>
  )
}

function Scene({ satellites, prediction }: any) {
  // Force show trajectories for HIGH_RISK and CAUTION
  const showTrajectories = prediction && (prediction.risk_level === 'HIGH_RISK' || prediction.risk_level === 'CAUTION')
  const sat1Color = prediction?.risk_level === 'HIGH_RISK' ? '#ef4444' : prediction?.risk_level === 'CAUTION' ? '#f59e0b' : '#00d9ff'
  const sat2Color = prediction?.risk_level === 'HIGH_RISK' ? '#ef4444' : prediction?.risk_level === 'CAUTION' ? '#f59e0b' : '#ec4899'
  
  useEffect(() => {
    console.log('🎯 Scene render - showTrajectories:', showTrajectories)
    console.log('🎯 Scene render - prediction:', prediction)
    console.log('🎯 Scene render - prediction.risk_level:', prediction?.risk_level)
    console.log('🎯 Scene render - satellites:', satellites)
  }, [showTrajectories, prediction, satellites])
  
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <Stars radius={300} depth={60} count={5000} factor={7} />
      
      <Earth />
      
      {/* Collision warning ring */}
      {prediction && <CollisionWarning prediction={prediction} />}
      
      {/* Satellite 1 */}
      {satellites.length > 0 && satellites[0] && (
        <>
          <Satellite
            satellite={satellites[0]}
            color={sat1Color}
            label={satellites[0]?.name || 'SAT-A'}
          />
          {prediction && (
            <TrajectoryPath
              satellite={satellites[0]}
              color={sat1Color}
            />
          )}
        </>
      )}
      
      {/* Satellite 2 */}
      {satellites.length > 1 && satellites[1] && (
        <>
          <Satellite
            satellite={satellites[1]}
            color={sat2Color}
            label={satellites[1]?.name || 'SAT-B'}
          />
          {prediction && (
            <TrajectoryPath
              satellite={satellites[1]}
              color={sat2Color}
            />
          )}
        </>
      )}
      
      {/* Collision point marker */}
      {showTrajectories && (
        <CollisionPoint satellites={satellites} prediction={prediction} />
      )}
      
      <OrbitControls
        enableZoom={true}
        enablePan={true}
        enableRotate={true}
        minDistance={2}
        maxDistance={15}
      />
    </>
  )
}

export default function OrbitCanvas({ satellites, prediction }: { satellites: any[], prediction?: any }) {
  console.log('🌍 OrbitCanvas render - prediction:', prediction)
  
  return (
    <div className="w-full h-full bg-black relative">
      <Canvas camera={{ position: [5, 3, 5], fov: 60 }} key={prediction?.risk_level || 'no-prediction'}>
        <Scene satellites={satellites} prediction={prediction} />
      </Canvas>
      {prediction && prediction.risk_level !== 'SAFE' && (
        <div className="absolute top-4 right-4 bg-red-500/90 text-white px-4 py-2 rounded-lg font-bold animate-pulse">
          ⚠️ {prediction.risk_level.replace('_', ' ')} DETECTED
        </div>
      )}
      {/* Debug indicator */}
      <div className="absolute top-4 left-4 bg-blue-500/90 text-white px-3 py-1 rounded text-xs">
        Trajectories: {prediction && prediction.risk_level !== 'SAFE' ? 'ACTIVE' : 'OFF'}
      </div>
    </div>
  )
}
