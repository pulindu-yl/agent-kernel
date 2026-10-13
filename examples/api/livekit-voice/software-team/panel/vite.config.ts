import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The token comes from the example's server, which also serves the built page at http://localhost:8000.
    proxy: {
      "/token": "http://localhost:8000",
      "/project": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // livekit-client alone is about 750 kB (200 kB gzipped); one chunk is fine for a demo page.
    chunkSizeWarningLimit: 1000,
  },
});
