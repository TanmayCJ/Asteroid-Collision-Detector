/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.externals.push({
      'three-mesh-bvh': 'three-mesh-bvh'
    })
    return config
  }
}

module.exports = nextConfig
