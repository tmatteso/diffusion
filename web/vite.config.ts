import stylex from "@stylexjs/unplugin";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

/**
 * StyleX's Vite plugin, minus its dev-server hook when running under Vitest.
 *
 * The hook starts an interval timer that it only clears when Vite's HTTP
 * server closes. Vitest runs Vite without an HTTP server, so the timer would
 * keep the process alive after the tests finish. Tests still get the full
 * StyleX transform; they only lose the dev CSS endpoint and hot reload.
 *
 * @returns The StyleX plugin to register with Vite.
 */
function stylexPlugin(): Plugin<unknown> {
  const plugin: Plugin<unknown> = stylex.vite();
  if (process.env["VITEST"] !== undefined) {
    delete plugin.configureServer;
  }
  return plugin;
}

export default defineConfig({
  plugins: [stylexPlugin(), react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/testSetup.ts"],
  },
});
