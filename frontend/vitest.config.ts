import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    include: ["**/*.test.{ts,tsx}"],
    coverage: { include: ["lib/**", "components/**"], reporter: ["text"] },
  },
  resolve: { alias: { "@": __dirname } },
});
