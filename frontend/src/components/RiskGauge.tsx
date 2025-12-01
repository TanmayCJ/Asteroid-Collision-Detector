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

  const getRiskPercentage = (level: string) => {
    switch (level) {
      case 'SAFE': return 20
      case 'CAUTION': return 60
      case 'HIGH_RISK': return 95
      default: return 0
    }
  }

  const riskColor = getRiskColor(prediction.risk_level)
  const riskPercent = getRiskPercentage(prediction.risk_level)

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
          {prediction.risk_level.replace('_', ' ')}
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
