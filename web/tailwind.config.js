/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cork: {
          light: "#c39769",
          DEFAULT: "#b08355",
          dark: "#8a5a34",
        },
        note: {
          yellow: "#ffe98a",
          yellowEdge: "#e6cc66",
          mint: "#c9f0dc",
          mintEdge: "#9ccfb3",
          line: "#fffdf3",
        },
        ink: {
          DEFAULT: "#4a3b26",
          soft: "#7a684c",
        },
      },
      fontFamily: {
        hand: ['"Ma Shan Zheng"', "cursive"],
      },
      boxShadow: {
        note: "0 10px 20px -8px rgba(74, 49, 20, 0.45), 0 3px 6px rgba(74,49,20,.25)",
        "note-sm": "0 5px 12px -6px rgba(74,49,20,.45)",
        "note-lg": "0 24px 60px -12px rgba(74, 49, 20, 0.55), 0 8px 16px rgba(74,49,20,.3)",
        pin: "0 3px 5px rgba(74,49,20,.4), inset 0 1px 1px rgba(255,255,255,.6)",
      },
    },
  },
  plugins: [],
};
