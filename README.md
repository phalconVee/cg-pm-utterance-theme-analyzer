# TT Labs UTheme Analyzer

**Utterance Theme Analyzer for PMs** — upload a CSV of customer utterances, get axial-coded thematic analysis with pain points, cross-topic patterns, and actionable insights in seconds.

Built for Product Managers who don't want to wait for analyst cycles to understand what their AI agents' users are saying.

## What it does

1. **Upload** a CSV with `utterance` and `topic_id` columns
2. **Claude performs qualitative analysis** — open coding → axial coding → selective coding
3. **Get a dashboard** with themes, sentiment, pain points, cross-topic patterns, and recommendations
4. **Export** as Markdown report, CSV, or raw JSON

Large datasets are automatically batched and merged — no size limits.

## Quick Start

### Prerequisites

- **Node.js 18+** — check with `node --version`
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com/settings/keys)

### Setup (2 minutes)

```bash
# 1. Install dependencies
npm install

# 2. Add your API key
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY

# 3. Run it
npm run dev
```

Open **http://localhost:5173** — that's it.

A `sample-data.csv` is included so you can test immediately.

## Architecture

```
<folder>/
├── server/proxy.js      # Thin Express proxy — keeps API key server-side
├── src/
│   ├── main.jsx         # React entry point
│   └── App.jsx          # Full application (single file)
├── sample-data.csv      # Test data — 30 utterances across 5 topics
├── .env.example         # API key template
├── package.json
└── vite.config.js       # Dev server + proxy config
```

## Export Formats

| Format   | Best for                                      |
|----------|-----------------------------------------------|
| Markdown | Sharing in Confluence, Notion, Slack, or email|
| CSV      | Further analysis in Sheets/Excel              |
| JSON     | Feeding into pipelines or tracking over time  |

## Deploying

For internal team use, the simplest path:

1. Deploy the Express proxy to any Node host (Railway, Render, internal VM)
2. `npm run build` produces static files in `dist/` — serve from anywhere
3. Set `ANTHROPIC_API_KEY` as an env var on the server

No database needed. All processing is stateless.

## License

MIT
