'use client'

import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'

export default function AlertCard({ prediction }: any) {
  const getIcon = (level: string) => {
    switch (level) {
      case 'SAFE':
        return <CheckCircle className="w-6 h-6 text-green-500" />
      case 'CAUTION':
        return <AlertCircle className="w-6 h-6 text-yellow-500" />
      case 'HIGH_RISK':
        return <AlertTriangle className="w-6 h-6 text-red-500" />
      default:
        return null
    }
  }

  const getMessage = (level: string) => {
    switch (level) {
      case 'SAFE':
        return 'No immediate collision risk detected. Satellites maintain safe separation distance.'
      case 'CAUTION':
        return 'Moderate risk detected. Close approach predicted within 24 hours. Recommend monitoring.'
      case 'HIGH_RISK':
        return '⚠️ HIGH RISK: Collision possible! Immediate action recommended. Consider orbital maneuver.'
      default:
        return 'Analyzing...'
    }
  }

  const getBgColor = (level: string) => {
    switch (level) {
      case 'SAFE': return 'bg-green-900/20 border-green-500'
      case 'CAUTION': return 'bg-yellow-900/20 border-yellow-500'
      case 'HIGH_RISK': return 'bg-red-900/20 border-red-500 animate-pulse'
      default: return 'bg-gray-900/20 border-gray-500'
    }
  }

  return (
    <div className={`glass-panel rounded-xl p-6 border-l-4 ${getBgColor(prediction.risk_level)}`}>
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          {getIcon(prediction.risk_level)}
        </div>
        
        <div className="flex-1">
          <h3 className="font-bold text-lg mb-2">
            Alert: {prediction.satellite_a} ↔ {prediction.satellite_b}
          </h3>
          
          <p className="text-sm text-gray-300 mb-4">
            {getMessage(prediction.risk_level)}
          </p>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-gray-400">Relative Velocity</p>
              <p className="font-bold">
                {(prediction.relative_velocity_kmps * 1000).toFixed(2)} m/s
              </p>
            </div>
            <div>
              <p className="text-gray-400">Horizon</p>
              <p className="font-bold">{prediction.prediction_horizon_hours}h</p>
            </div>
          </div>

          {prediction.risk_level === 'HIGH_RISK' && (
            <button className="mt-4 w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition-colors">
              Plan Avoidance Maneuver
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
