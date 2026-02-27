import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json({ limit: "5mb" }));

// Health check
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", hasKey: !!process.env.ANTHROPIC_API_KEY });
});

// Proxy to Anthropic API
app.post("/api/analyze", async (req, res) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      error: "ANTHROPIC_API_KEY not set. Add it to your .env file.",
    });
  }

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: req.body.model || "claude-sonnet-4-20250514",
        max_tokens: req.body.max_tokens || 8096,
        messages: req.body.messages,
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      return res.status(response.status).json({
        error: errData.error?.message || `Anthropic API error: ${response.status}`,
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Proxy error:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n  🔧 Axial Coder proxy running at http://localhost:${PORT}`);
  console.log(`  ${process.env.ANTHROPIC_API_KEY ? "✅ API key loaded" : "❌ No API key — add ANTHROPIC_API_KEY to .env"}\n`);
});
