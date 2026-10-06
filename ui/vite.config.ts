import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // 5174 (strict) is the app's canonical dev URL; the widget dev harness in
  // frontend/ uses Vite's default 5173. Keycloak's dev client accepts any
  // http://localhost:* redirect.
  server: { port: 5174, strictPort: true },
  preview: { port: 5174, strictPort: true },
});
