import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 通过 IDE 预览隧道访问时，浏览器所在机器访问不到沙箱的 localhost:2024，
// 因此把 /lg 前缀同源代理到 Vite 进程所在沙箱内的 langgraph dev（含 SSE）。
// 本地直连开发时默认走该代理即可；需要跨机后端可用 .env 的 VITE_API_URL 覆盖。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 仅绑回环：通过 IDE 端口转发/预览访问，避免广播沙箱内网 IP
    host: "127.0.0.1",
    proxy: {
      "/lg": {
        target: "http://localhost:2024",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/lg/, ""),
      },
      // 调试设置（admin_server 挂在 langgraph dev 根路径，无需 rewrite）
      "/admin": {
        target: "http://localhost:2024",
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // 稳定的第三方分包：升级单块依赖时其余 chunk 缓存仍可复用
        manualChunks: {
          react: ["react", "react-dom"],
          sdk: ["@langchain/langgraph-sdk"],
          markdown: [
            "react-markdown",
            "remark-gfm",
            "rehype-highlight",
            "lowlight",
          ],
        },
      },
    },
  },
});
