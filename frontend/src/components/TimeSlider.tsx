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
      {/* Risk amplitude graph above timeline */}
      <div className="relative h-32 bg-space-800/50 rounded-lg p-4">
        <svg className="w-full h-full" viewBox="0 0 1000 120">
          <defs>
            {/* Gradient for high risk areas */}
            <linearGradient id="riskGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.1" />
            </linearGradient>
            <linearGradient id="cautionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.1" />
            </linearGradient>
            <linearGradient id="safeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.1" />
            </linearGradient>
          </defs>
          
          {/* Grid lines */}
          {[20, 40, 60, 80, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="1000"
              y2={y}
              stroke="#2d3561"
              strokeWidth="1"
              opacity="0.3"
            />
          ))}
          
          {/* Risk amplitude curve - inverted so high risk = high amplitude */}
          <path
            d={(() => {
              let pathData = 'M 0 120'
              for (let i = 0; i <= 1000; i += 20) {
                const hour = (i / 1000) * hours
                const dist = getDistanceAtTime(hour)
                // Invert: closer distance = higher amplitude (lower y value)
                // Scale: 0-5km -> 0-100px amplitude
                const amplitude = Math.max(0, Math.min(100, (25 - dist) * 4))
                pathData += ` L ${i} ${120 - amplitude}`
              }
              pathData += ' L 1000 120 Z'
              return pathData
            })()}
            fill="url(#riskGradient)"
            opacity="0.9"
          />
          
          {/* Stroke line for the curve */}
          <path
            d={(() => {
              let pathData = 'M 0'
              for (let i = 0; i <= 1000; i += 20) {
                const hour = (i / 1000) * hours
                const dist = getDistanceAtTime(hour)
                const amplitude = Math.max(0, Math.min(100, (25 - dist) * 4))
                pathData += ` ${i === 0 ? '' : 'L'} ${i} ${120 - amplitude}`
              }
              return pathData
            })()}
            fill="none"
            stroke="#ef4444"
            strokeWidth="2"
          />
          
          {/* Current time marker - vertical line */}
          <line
            x1={(timeIndex / hours) * 1000}
            y1="0"
            x2={(timeIndex / hours) * 1000}
            y2="120"
            stroke="#00d9ff"
            strokeWidth="2"
            strokeDasharray="4 4"
          />
          
          {/* Current risk amplitude marker */}
          <circle
            cx={(timeIndex / hours) * 1000}
            cy={120 - Math.max(0, Math.min(100, (25 - currentDistance) * 4))}
            r="6"
            fill="#00d9ff"
            stroke="#ffffff"
            strokeWidth="2"
          >
            <animate
              attributeName="r"
              values="6;8;6"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </circle>
          
          {/* Risk zone labels */}
          <text x="10" y="15" fill="#ef4444" fontSize="10" opacity="0.7">HIGH RISK</text>
          <text x="10" y="65" fill="#f59e0b" fontSize="10" opacity="0.7">CAUTION</text>
          <text x="10" y="110" fill="#10b981" fontSize="10" opacity="0.7">SAFE</text>
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
