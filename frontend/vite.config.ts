import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: "src/widget.tsx",
      name: "AssistantWidget",
      fileName: "assistant-widget",
      formats: ["es", "umd"],
    },
  },
});
