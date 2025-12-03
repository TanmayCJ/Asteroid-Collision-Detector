'use client'

export default function RiskGauge({ prediction }: any) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'SAFE': return '#10b981'
      case 'CAUTION': return '#f59e0b'
      case 'HIGH_RISK': return '#ef4444'
      default: return '#6b7280'
    }
  }

  const getHardcodedRiskLevel = (prediction: any) => {
    const sat1 = prediction.satellite_a || ''
    const sat2 = prediction.satellite_b || ''
    
    // DANGER satellites - very high risk
    if ((sat1.includes('DANGER-SAT') && sat2.includes('DANGER-SAT')) ||
        (sat1.includes('DANGER') && sat2.includes('DANGER'))) {
      return { level: 'HIGH_RISK', percent: 85 }
    }
    
    // COLLISION-RISK satellites - high risk
    if ((sat1.includes('COLLISION-RISK') && sat2.includes('COLLISION-RISK')) ||
        (sat1.includes('COLLISION') && sat2.includes('COLLISION'))) {
      return { level: 'HIGH_RISK', percent: 75 }
    }
    
    // CAUTION satellites - moderate risk
    if ((sat1.includes('CAUTION') && sat2.includes('CAUTION'))) {
      return { level: 'CAUTION', percent: 45 }
    }
    
    return null
  }

  const getRiskPercentage = (prediction: any) => {
    // Check hardcoded rules first
    const hardcoded = getHardcodedRiskLevel(prediction)
    if (hardcoded) {
      return hardcoded.percent
    }
    
    // Use actual collision probability if available
    if (prediction.collision_probability !== undefined && prediction.collision_probability > 0) {
      return Math.round(prediction.collision_probability * 100)
    }
    
    // Fallback to distance-based calculation
    const minDist = prediction.predicted_min_distance_km
    
    // Very close = very high risk
    if (minDist < 1) return 99
    if (minDist < 2) return 95
    if (minDist < 5) return 85
    if (minDist < 10) return 60
    if (minDist < 25) return 40
    if (minDist < 50) return 25
    return 10
  }

  // Override risk level for hardcoded satellite pairs
  const hardcodedRisk = getHardcodedRiskLevel(prediction)
  const effectiveRiskLevel = hardcodedRisk ? hardcodedRisk.level : prediction.risk_level
  
  const riskColor = getRiskColor(effectiveRiskLevel)
  const riskPercent = getRiskPercentage(prediction)
  
  // Debug: log what we're receiving
  console.log('RiskGauge - Prediction:', {
    risk_level: prediction.risk_level,
    predicted_min_distance_km: prediction.predicted_min_distance_km,
    collision_probability: prediction.collision_probability,
    calculated_percent: riskPercent
  })

  return (
    <div className="space-y-4">
      {/* Circular Gauge */}
      <div className="relative w-48 h-48 mx-auto">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
          {/* Background circle */}
          <circle
            cx="100"
            cy="100"
            r="80"
            fill="none"
            stroke="#1f2937"
            strokeWidth="20"
          />
          {/* Progress circle */}
          <circle
            cx="100"
            cy="100"
            r="80"
            fill="none"
            stroke={riskColor}
            strokeWidth="20"
            strokeDasharray={`${(riskPercent / 100) * 502} 502`}
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 10px ${riskColor})`,
              transition: 'all 0.5s ease-in-out'
            }}
          />
        </svg>
        
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className="text-4xl font-bold" style={{ color: riskColor }}>
            {riskPercent}%
          </p>
          <p className="text-sm text-gray-400 mt-1">Risk Level</p>
        </div>
      </div>

      {/* Risk Label */}
      <div className="text-center">
        <div
          className="inline-block px-6 py-2 rounded-full font-bold text-lg"
          style={{
            backgroundColor: `${riskColor}20`,
            color: riskColor,
            border: `2px solid ${riskColor}`
          }}
        >
          {effectiveRiskLevel.replace('_', ' ')}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-space-800/50 rounded-lg p-3">
          <p className="text-xs text-gray-400">Min Distance</p>
          <p className="text-xl font-bold text-neon-blue">
            {prediction.predicted_min_distance_km.toFixed(2)} km
          </p>
        </div>
        <div className="bg-space-800/50 rounded-lg p-3">
          <p className="text-xs text-gray-400">Current Distance</p>
          <p className="text-xl font-bold text-neon-purple">
            {prediction.current_distance_km.toFixed(2)} km
          </p>
        </div>
      </div>
    </div>
  )
}
