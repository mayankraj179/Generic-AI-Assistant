import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run build:embed` → dist-embed/bitwise-assist.js: the launcher as ONE
// classic script (IIFE, React included, CSS inlined into the shadow root),
// for a <script src> tag on any host page. Separate from the standalone
// app's build (vite.config.ts → dist/).
export default defineConfig({
  plugins: [react()],
  // Library mode doesn't replace process.env; React needs it for its prod build.
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  publicDir: false,
  build: {
    outDir: "dist-embed",
    // ES2022 keeps class fields native; lower targets make esbuild emit
    // helper vars outside the IIFE, i.e. globals on the host page.
    target: "es2022",
    emptyOutDir: true,
    lib: {
      entry: "src/embed.tsx",
      name: "BitwiseAssist",
      formats: ["iife"],
      fileName: () => "bitwise-assist.js",
    },
  },
});
