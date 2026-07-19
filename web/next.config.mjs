/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // Keep edge runtime for chat / mcp routes; per-route `export const
    // runtime = 'edge'` is the source of truth.
  },
  images: {
    // Leaflet tiles proxy if we ever self-host tiles; default OSM tiles
    // come from tile.openstreetmap.org and are public.
    remotePatterns: [
      { protocol: "https", hostname: "*.tile.openstreetmap.org" },
    ],
  },
  webpack: (config) => {
    // Leaflet expects `window`, so we lazy-load it from client-only
    // components (dynamic import with ssr:false). This shim silences
    // a warning on the server build.
    config.resolve.fallback = { ...config.resolve.fallback, fs: false };
    return config;
  },
};

export default nextConfig;
