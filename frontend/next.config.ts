import type { NextConfig } from "next";

// The browser always calls same-origin /api/*; Next proxies it to the FastAPI service.
// No CORS, and the backend URL stays a server-side setting (API_URL).
const apiUrl = process.env.API_URL ?? "http://localhost:8000";

const config: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
};

export default config;
