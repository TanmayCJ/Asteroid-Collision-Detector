'use client'

import { useState, useEffect } from 'react'
import OrbitCanvas from '@/components/OrbitCanvas'
import RiskGauge from '@/components/RiskGauge'
import AlertCard from '@/components/AlertCard'
import TimeSlider from '@/components/TimeSlider'
import { fetchPrediction, fetchSatellites } from '@/lib/api'

export default function Dashboard() {
  const [satellites, setSatellites] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [selectedPair, setSelectedPair] = useState({ satA: null, satB: null })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadSatellites()
  }, [])

  const loadSatellites = async () => {
    try {
      const data = await fetchSatellites()
      setSatellites(data)
      if (data.length >= 2) {
        setSelectedPair({ satA: data[0], satB: data[1] })
      }
    } catch (error) {
      console.error('Failed to load satellites:', error)
    }
  }

  const runPrediction = async () => {
    if (!selectedPair.satA || !selectedPair.satB) return
    
    setLoading(true)
    try {
      const result = await fetchPrediction(
        selectedPair.satA.norad_id,
        selectedPair.satB.norad_id
      )
      setPrediction(result)
    } catch (error) {
      console.error('Prediction failed:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-space-900 text-white">
      {/* Header */}
      <header className="border-b border-space-700 bg-space-800/50 backdrop-blur">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple flex items-center justify-center">
                <span className="text-2xl">🛰️</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold glow-text">AstroGuard</h1>
                <p className="text-sm text-gray-400">Satellite Collision Predictor</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right text-sm">
                <p className="text-gray-400">Tracking</p>
                <p className="font-bold text-neon-blue">{satellites.length} Satellites</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Satellite Selection */}
            <div className="glass-panel rounded-xl p-6">
              <h2 className="text-xl font-bold mb-4 text-neon-blue">
                Select Satellites
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Satellite A
                  </label>
                  <select
                    className="w-full bg-space-800 border border-space-600 rounded-lg px-4 py-2 text-white"
                    onChange={(e) => {
                      const sat = satellites.find(s => s.norad_id === e.target.value)
                      setSelectedPair({ ...selectedPair, satA: sat })
                    }}
                    value={selectedPair.satA?.norad_id || ''}
                  >
                    <option value="">Select satellite...</option>
                    {satellites.map((sat) => (
                      <option key={sat.norad_id} value={sat.norad_id}>
                        {sat.name} ({sat.orbit_regime})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Satellite B
                  </label>
                  <select
                    className="w-full bg-space-800 border border-space-600 rounded-lg px-4 py-2 text-white"
                    onChange={(e) => {
                      const sat = satellites.find(s => s.norad_id === e.target.value)
                      setSelectedPair({ ...selectedPair, satB: sat })
                    }}
                    value={selectedPair.satB?.norad_id || ''}
                  >
                    <option value="">Select satellite...</option>
                    {satellites.map((sat) => (
                      <option key={sat.norad_id} value={sat.norad_id}>
                        {sat.name} ({sat.orbit_regime})
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  onClick={runPrediction}
                  disabled={loading || !selectedPair.satA || !selectedPair.satB}
                  className="w-full bg-gradient-to-r from-neon-blue to-neon-purple rounded-lg px-6 py-3 font-bold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-neon-blue/50 transition-all"
                >
                  {loading ? 'Analyzing...' : 'Run Prediction'}
                </button>
              </div>
            </div>

            {/* Risk Gauge */}
            {prediction && (
              <div className="glass-panel rounded-xl p-6">
                <h2 className="text-xl font-bold mb-4 text-neon-purple">
                  Collision Risk
                </h2>
                <RiskGauge prediction={prediction} />
              </div>
            )}

            {/* Alert Card */}
            {prediction && (
              <AlertCard prediction={prediction} />
            )}
          </div>

          {/* Center Panel - 3D Visualization */}
          <div className="lg:col-span-2 space-y-6">
            <div className="glass-panel rounded-xl overflow-hidden" style={{ height: '600px' }}>
              <OrbitCanvas 
                satellites={selectedPair.satA && selectedPair.satB ? [selectedPair.satA, selectedPair.satB] : []} 
                prediction={prediction}
              />
            </div>

            {/* Time Slider */}
            {prediction && (
              <div className="glass-panel rounded-xl p-6">
                <h2 className="text-xl font-bold mb-4 text-neon-green">
                  Timeline Analysis
                </h2>
                <TimeSlider prediction={prediction} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
