import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AstroGuard | Satellite Collision Predictor',
  description: 'ML-powered satellite collision prediction and visualization',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
