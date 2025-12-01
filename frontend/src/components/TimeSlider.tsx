'use client'

import { useState } from 'react'

export default function TimeSlider({ prediction }: any) {
  const [timeIndex, setTimeIndex] = useState(0)
  const hours = 24

  const getDistanceAtTime = (hour: number) => {
    // Simulate distance change over time
    const minDist = prediction.predicted_min_distance_km
    const currentDist = prediction.current_distance_km
    const closestApproach = hours / 2 // Assume closest approach at 12h
    
    // Parabolic approach
    const t = hour / hours
    const approachFactor = Math.abs(t - 0.5) * 2
    return minDist + (currentDist - minDist) * approachFactor
  }

  const currentDistance = getDistanceAtTime(timeIndex)
  const getRiskAtDistance = (dist: number) => {
    if (dist < 5) return 'HIGH_RISK'
    if (dist < 25) return 'CAUTION'
    return 'SAFE'
  }

  const currentRisk = getRiskAtDistance(currentDistance)

  return (
    <div className="space-y-6">
      {/* Timeline visualization */}
      <div className="relative h-24 bg-space-800/50 rounded-lg p-4">
        {/* Distance line */}
        <svg className="w-full h-full" viewBox="0 0 1000 100" preserveAspectRatio="none">
          <defs>
            <linearGradient id="distanceGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="50%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
          
          {/* Distance curve */}
          <path
            d={`M 0 ${100 - getDistanceAtTime(0) * 2} 
                L 250 ${100 - getDistanceAtTime(6) * 2}
                L 500 ${100 - getDistanceAtTime(12) * 2}
                L 750 ${100 - getDistanceAtTime(18) * 2}
                L 1000 ${100 - getDistanceAtTime(24) * 2}`}
            fill="none"
            stroke="url(#distanceGradient)"
            strokeWidth="3"
          />
          
          {/* Current time marker */}
          <circle
            cx={(timeIndex / hours) * 1000}
            cy={100 - currentDistance * 2}
            r="8"
            fill="#00d9ff"
            stroke="#ffffff"
            strokeWidth="2"
          />
        </svg>
      </div>

      {/* Time slider */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm text-gray-400">
          <span>Now</span>
          <span className="font-bold text-white">T+{timeIndex}h</span>
          <span>+24h</span>
        </div>
        
        <input
          type="range"
          min="0"
          max={hours}
          value={timeIndex}
          onChange={(e) => setTimeIndex(Number(e.target.value))}
          className="w-full h-2 bg-space-700 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #00d9ff 0%, #00d9ff ${(timeIndex/hours)*100}%, #2d3561 ${(timeIndex/hours)*100}%, #2d3561 100%)`
          }}
        />
      </div>

      {/* Current state display */}
      <div className="grid grid-cols-3 gap-4">
        <div className="glass-panel rounded-lg p-4 text-center">
          <p className="text-xs text-gray-400 mb-1">Time</p>
          <p className="text-lg font-bold text-neon-blue">+{timeIndex}h</p>
        </div>
        
        <div className="glass-panel rounded-lg p-4 text-center">
          <p className="text-xs text-gray-400 mb-1">Distance</p>
          <p className="text-lg font-bold text-neon-purple">{currentDistance.toFixed(1)} km</p>
        </div>
        
        <div className="glass-panel rounded-lg p-4 text-center">
          <p className="text-xs text-gray-400 mb-1">Risk</p>
          <p className={`text-lg font-bold ${
            currentRisk === 'SAFE' ? 'text-green-500' :
            currentRisk === 'CAUTION' ? 'text-yellow-500' :
            'text-red-500'
          }`}>
            {currentRisk.replace('_', ' ')}
          </p>
        </div>
      </div>
    </div>
  )
}
