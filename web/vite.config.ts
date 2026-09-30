import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The production build goes into the Python package, which serves it at "/".
// In development, `pnpm dev` proxies API calls to `sparring serve` on :8000.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/sparring/web/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { include: ["src/**/*.test.ts"] },
});
