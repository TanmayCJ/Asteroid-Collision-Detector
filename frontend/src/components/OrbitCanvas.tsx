'use client'

import { useRef, useEffect } from 'react'
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

function Satellite({ position, color, label }: any) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01
    }
  })

  // Calculate distance from origin (Earth center) to satellite position
  const orbitRadius = Math.sqrt(
    position[0] * position[0] + 
    position[1] * position[1] + 
    position[2] * position[2]
  )

  return (
    <group position={position}>
      <Sphere ref={meshRef} args={[0.05, 16, 16]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
        />
      </Sphere>
      {/* Orbit ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[orbitRadius, 0.005, 16, 100]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
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

function Scene({ satellites }: any) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <Stars radius={300} depth={60} count={5000} factor={7} />
      
      <Earth />
      
      {satellites.length > 0 && (
        <>
          <Satellite
            position={[2.5, 0.5, 1]}
            color="#00d9ff"
            label={satellites[0]?.name || 'SAT-A'}
          />
          {satellites.length > 1 && (
            <Satellite
              position={[2.8, -0.3, 0.8]}
              color="#ec4899"
              label={satellites[1]?.name || 'SAT-B'}
            />
          )}
        </>
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

export default function OrbitCanvas({ satellites }: { satellites: any[] }) {
  return (
    <div className="w-full h-full bg-black">
      <Canvas camera={{ position: [5, 3, 5], fov: 60 }}>
        <Scene satellites={satellites} />
      </Canvas>
    </div>
  )
}
