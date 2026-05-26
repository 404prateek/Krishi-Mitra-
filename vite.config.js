import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Move dep-optimization cache outside OneDrive to prevent sync-lock hangs
  cacheDir: 'C:/tmp/krishi-vite-cache',
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    host: true,
    // OneDrive / network-drive fix: use polling instead of native fs.watch
    // which gets stuck in an infinite re-transform loop on synced drives
    watch: {
      usePolling: true,
      interval: 1000,
    },
  },
  preview: {
    port: 3000,
    host: true,
  },
})