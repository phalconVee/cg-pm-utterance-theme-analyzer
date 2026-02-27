import { useState, useCallback, useRef, useEffect } from "react";
import Papa from "papaparse";

// ─── Color System ───────────────────────────────────────────────────────────
const PALETTE = {
  bg: "#0B0F1A",
  surface: "#121828",
  surfaceHover: "#1A2236",
  border: "#1E2A42",
  borderFocus: "#3B82F6",
  text: "#E2E8F0",
  textMuted: "#64748B",
  textDim: "#475569",
  accent: "#3B82F6",
  accentSoft: "rgba(59,130,246,0.12)",
  accentGlow: "rgba(59,130,246,0.25)",
  success: "#10B981",
  successSoft: "rgba(16,185,129,0.12)",
  warning: "#F59E0B",
  warningSoft: "rgba(245,158,11,0.12)",
  error: "#EF4444",
  errorSoft: "rgba(239,68,68,0.12)",
  purple: "#8B5CF6",
  purpleSoft: "rgba(139,92,246,0.12)",
  cyan: "#06B6D4",
  cyanSoft: "rgba(6,182,212,0.12)",
};

const THEME_COLORS = [
  "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
  "#06B6D4", "#EC4899", "#F97316", "#14B8A6", "#A855F7",
  "#6366F1", "#84CC16", "#E11D48", "#0EA5E9", "#D946EF",
];

// ─── Utility ────────────────────────────────────────────────────────────────
function truncate(str, len = 80) {
  if (!str) return "";
  return str.length > len ? str.slice(0, len) + "…" : str;
}

// ─── Constants ──────────────────────────────────────────────────────────────
const BATCH_SIZE = 40; // utterances per batch
const OUTPUT_SCHEMA = `{
  "themes": [{"name":"str","description":"str","sentiment":"positive|negative|neutral|mixed","frequency":0,"severity":"low|medium|high","topic_ids":["str"],"example_utterances":["str"],"codes":["str"]}],
  "cross_topic_patterns": [{"pattern":"str","topics_affected":["str"],"implication":"str"}],
  "pain_points": [{"issue":"str","severity":"low|medium|high","affected_topics":["str"],"recommendation":"str"}],
  "positive_signals": [{"signal":"str","topics":["str"],"strength":"weak|moderate|strong"}],
  "summary": {"total_utterances_analyzed":0,"total_topics":0,"core_theme":"str","overall_sentiment":"positive|negative|neutral|mixed","key_insight":"str","risk_areas":["str"],"opportunity_areas":["str"]}
}`;

// ─── API Helper ─────────────────────────────────────────────────────────────
async function callClaude(prompt, retries = 2) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8096,
          messages: [{ role: "user", content: prompt }],
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || errData.error?.message || `API error: ${response.status}`);
      }

      const apiData = await response.json();
      const text = (apiData.content || [])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");

      // Extract JSON from response — handle markdown fencing, preamble, etc.
      let jsonStr = text;
      const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (fenceMatch) jsonStr = fenceMatch[1];
      
      // Find the first { and last } to extract JSON object
      const firstBrace = jsonStr.indexOf("{");
      const lastBrace = jsonStr.lastIndexOf("}");
      if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        jsonStr = jsonStr.slice(firstBrace, lastBrace + 1);
      }

      return JSON.parse(jsonStr.trim());
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
    }
  }
}

// ─── Batch Utilities ────────────────────────────────────────────────────────
function chunkUtterances(utterances) {
  const batches = [];
  for (let i = 0; i < utterances.length; i += BATCH_SIZE) {
    batches.push(utterances.slice(i, i + BATCH_SIZE));
  }
  return batches;
}

function formatBatchData(utterances) {
  const grouped = {};
  utterances.forEach((u) => {
    const tid = u.topic_id || "unknown";
    if (!grouped[tid]) grouped[tid] = [];
    grouped[tid].push(u.utterance);
  });
  return Object.entries(grouped)
    .map(([tid, utts]) => `## Topic: ${tid}\n${utts.map((u, i) => `${i + 1}. "${u}"`).join("\n")}`)
    .join("\n\n");
}

// ─── Prompt Builders ────────────────────────────────────────────────────────
function buildBatchCodingPrompt(utterances) {
  return `You are an expert qualitative researcher. Perform open coding on these customer utterances from an AI agent.

For each group of utterances, identify:
- Codes (short labels for patterns)
- Sentiment per code
- Which topic_ids each code appears in
- Pain points or confusion signals
- Positive signals

Respond ONLY with a valid JSON object. No markdown, no backticks, no explanation before or after the JSON.

${OUTPUT_SCHEMA}

Keep descriptions concise (under 20 words each). Limit example_utterances to 2 per theme. Here are the utterances:

${formatBatchData(utterances)}`;
}

function buildSynthesisPrompt(batchResults, totalUtterances, totalTopics) {
  return `You are an expert qualitative researcher performing axial and selective coding. You have already completed open coding in batches. Now synthesize these batch results into a unified analysis.

Merge duplicate or overlapping themes. Identify cross-cutting patterns. Perform selective coding to find the core category.

Here are the batch results to synthesize:

${JSON.stringify(batchResults, null, 1)}

Total utterances analyzed: ${totalUtterances}
Total topics: ${totalTopics}

Respond ONLY with a valid JSON object. No markdown, no backticks, no explanation before or after the JSON. Use this structure:

${OUTPUT_SCHEMA}

Rules:
- Merge similar themes (combine frequencies, union topic_ids and examples)
- Keep max 10 themes (most significant)
- Keep max 2 example_utterances per theme
- Descriptions under 25 words
- Ensure total_utterances_analyzed = ${totalUtterances} and total_topics = ${totalTopics}
- Provide an insightful core_theme and key_insight`;
}

// ─── Components ─────────────────────────────────────────────────────────────

function Badge({ children, color = PALETTE.accent, bg }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "2px 10px",
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.03em",
        color: color,
        background: bg || `${color}18`,
        border: `1px solid ${color}30`,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

function SentimentBadge({ sentiment }) {
  const map = {
    positive: { color: PALETTE.success, label: "Positive" },
    negative: { color: PALETTE.error, label: "Negative" },
    neutral: { color: PALETTE.textMuted, label: "Neutral" },
    mixed: { color: PALETTE.warning, label: "Mixed" },
  };
  const s = map[sentiment] || map.neutral;
  return <Badge color={s.color}>{s.label}</Badge>;
}

function SeverityBadge({ severity }) {
  const map = {
    low: { color: PALETTE.success, label: "Low" },
    medium: { color: PALETTE.warning, label: "Medium" },
    high: { color: PALETTE.error, label: "High" },
  };
  const s = map[severity] || map.low;
  return <Badge color={s.color}>{s.label}</Badge>;
}

function Card({ children, style, onClick, hoverable }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered && hoverable ? PALETTE.surfaceHover : PALETTE.surface,
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 12,
        padding: 20,
        transition: "all 0.2s ease",
        cursor: onClick ? "pointer" : "default",
        ...(hovered && hoverable
          ? { borderColor: PALETTE.borderFocus, transform: "translateY(-1px)" }
          : {}),
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function StatCard({ label, value, sub, icon, color = PALETTE.accent }) {
  return (
    <Card style={{ flex: "1 1 0", minWidth: 140 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: `${color}18`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 16,
          }}
        >
          {icon}
        </div>
        <span style={{ color: PALETTE.textMuted, fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: PALETTE.text, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: PALETTE.textDim, marginTop: 4 }}>{sub}</div>}
    </Card>
  );
}

function ProgressBar({ value, max, color = PALETTE.accent, height = 6 }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div style={{ width: "100%", height, borderRadius: height, background: `${color}15`, overflow: "hidden" }}>
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          borderRadius: height,
          background: `linear-gradient(90deg, ${color}, ${color}CC)`,
          transition: "width 0.6s cubic-bezier(.4,0,.2,1)",
        }}
      />
    </div>
  );
}

// ─── Upload Zone ────────────────────────────────────────────────────────────
function UploadZone({ onFileParsed }) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const parseFile = useCallback(
    (file) => {
      setError(null);
      if (!file) return;
      if (!file.name.match(/\.(csv|tsv|txt)$/i)) {
        setError("Please upload a .csv, .tsv, or .txt file");
        return;
      }
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const cols = results.meta.fields || [];
          if (cols.length === 0) {
            setError("Could not detect any columns. Check your file format.");
            return;
          }
          if (results.data.length === 0) {
            setError("File has headers but no data rows.");
            return;
          }
          onFileParsed({ columns: cols, rows: results.data, filename: file.name });
        },
        error: (err) => setError(`Parse error: ${err.message}`),
      });
    },
    [onFileParsed]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); parseFile(e.dataTransfer.files[0]); }}
      onClick={() => fileRef.current?.click()}
      style={{
        border: `2px dashed ${dragOver ? PALETTE.accent : PALETTE.border}`,
        borderRadius: 16,
        padding: "48px 32px",
        textAlign: "center",
        cursor: "pointer",
        background: dragOver ? PALETTE.accentSoft : "transparent",
        transition: "all 0.25s ease",
      }}
    >
      <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" style={{ display: "none" }} onChange={(e) => parseFile(e.target.files[0])} />
      <div style={{ fontSize: 40, marginBottom: 12 }}>📄</div>
      <div style={{ fontSize: 16, fontWeight: 600, color: PALETTE.text, marginBottom: 6 }}>
        Drop your CSV here, or click to browse
      </div>
      <div style={{ fontSize: 13, color: PALETTE.textMuted }}>
        Any CSV with customer utterances — you'll map columns in the next step
      </div>
      {error && (
        <div style={{ marginTop: 16, padding: "10px 16px", borderRadius: 8, background: PALETTE.errorSoft, color: PALETTE.error, fontSize: 13, fontWeight: 500 }}>
          {error}
        </div>
      )}
    </div>
  );
}

// ─── Column Auto-Detect ─────────────────────────────────────────────────────
const UTTERANCE_HINTS = ["utterance", "text", "message", "query", "input", "question", "comment", "customer_message", "user_input", "user_message", "prompt", "request", "content", "body", "description", "feedback"];
const TOPIC_HINTS = ["topic", "topic_id", "category", "subject", "intent", "label", "tag", "group", "type", "class", "domain", "area", "section", "skill", "agent", "flow"];

function scoreColumn(colName, hints) {
  const lower = colName.toLowerCase().replace(/[_\-\s]+/g, "");
  for (const hint of hints) {
    const h = hint.replace(/[_\-\s]+/g, "");
    if (lower === h) return 100;
    if (lower.includes(h) || h.includes(lower)) return 70;
  }
  return 0;
}

function autoDetectMapping(columns) {
  let bestUtterance = { col: "", score: 0 };
  let bestTopic = { col: "", score: 0 };

  for (const col of columns) {
    const uScore = scoreColumn(col, UTTERANCE_HINTS);
    const tScore = scoreColumn(col, TOPIC_HINTS);
    if (uScore > bestUtterance.score) bestUtterance = { col, score: uScore };
    if (tScore > bestTopic.score) bestTopic = { col, score: tScore };
  }

  // Don't map the same column to both fields
  if (bestUtterance.col && bestUtterance.col === bestTopic.col) {
    if (bestUtterance.score >= bestTopic.score) bestTopic = { col: "", score: 0 };
    else bestUtterance = { col: "", score: 0 };
  }

  return {
    utterance: bestUtterance.col,
    topic_id: bestTopic.col,
    autoDetected: { utterance: bestUtterance.score > 0, topic_id: bestTopic.score > 0 },
  };
}

// ─── Column Mapper ──────────────────────────────────────────────────────────
function ColumnMapper({ columns, rows, filename, onConfirm, onBack }) {
  const detected = autoDetectMapping(columns);
  const [mapping, setMapping] = useState({
    utterance: detected.utterance,
    topic_id: detected.topic_id,
  });

  const UNMAPPED = "__none__";

  const previewRows = rows.slice(0, 5);
  const utteranceCol = mapping.utterance;
  const topicCol = mapping.topic_id;
  const isValid = utteranceCol && utteranceCol !== UNMAPPED;

  const handleConfirm = () => {
    const parsed = rows
      .map((row) => ({
        utterance: (row[utteranceCol] || "").trim(),
        topic_id: topicCol && topicCol !== UNMAPPED ? (row[topicCol] || "unknown").trim() : "unknown",
      }))
      .filter((r) => r.utterance.length > 0);
    onConfirm(parsed, filename);
  };

  const selectStyle = (isSet, isAutoDetected) => ({
    width: "100%",
    padding: "10px 12px",
    borderRadius: 8,
    border: `1.5px solid ${isSet ? (isAutoDetected ? PALETTE.success : PALETTE.accent) : PALETTE.border}`,
    background: PALETTE.bg,
    color: isSet ? PALETTE.text : PALETTE.textDim,
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    outline: "none",
    appearance: "none",
    WebkitAppearance: "none",
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2364748B' d='M2 4l4 4 4-4'/%3E%3C/svg%3E")`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "right 12px center",
    paddingRight: 32,
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: PALETTE.text }}>Map your columns</div>
            <div style={{ fontSize: 12, color: PALETTE.textMuted, marginTop: 2 }}>
              {filename} · {columns.length} columns · {rows.length} rows
            </div>
          </div>
          <button
            onClick={onBack}
            style={{
              padding: "6px 14px", borderRadius: 7, border: `1px solid ${PALETTE.border}`,
              background: "transparent", color: PALETTE.textMuted, fontSize: 12, fontWeight: 500, cursor: "pointer",
            }}
          >
            ← Back
          </button>
        </div>

        {/* Mapping fields */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
          {/* Utterance mapping */}
          <div style={{ flex: "1 1 220px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: PALETTE.text }}>Utterance column</span>
              <span style={{ fontSize: 10, color: PALETTE.error }}>required</span>
              {detected.autoDetected.utterance && mapping.utterance === detected.utterance && (
                <Badge color={PALETTE.success}>auto-detected</Badge>
              )}
            </div>
            <select
              value={mapping.utterance || UNMAPPED}
              onChange={(e) => setMapping((m) => ({ ...m, utterance: e.target.value === UNMAPPED ? "" : e.target.value }))}
              style={selectStyle(!!mapping.utterance, detected.autoDetected.utterance && mapping.utterance === detected.utterance)}
            >
              <option value={UNMAPPED}>Select column…</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <div style={{ fontSize: 11, color: PALETTE.textDim, marginTop: 4 }}>
              The column containing customer messages or queries
            </div>
          </div>

          {/* Topic mapping */}
          <div style={{ flex: "1 1 220px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: PALETTE.text }}>Topic / Category column</span>
              <span style={{ fontSize: 10, color: PALETTE.textDim }}>optional</span>
              {detected.autoDetected.topic_id && mapping.topic_id === detected.topic_id && (
                <Badge color={PALETTE.success}>auto-detected</Badge>
              )}
            </div>
            <select
              value={mapping.topic_id || UNMAPPED}
              onChange={(e) => setMapping((m) => ({ ...m, topic_id: e.target.value === UNMAPPED ? "" : e.target.value }))}
              style={selectStyle(!!mapping.topic_id, detected.autoDetected.topic_id && mapping.topic_id === detected.topic_id)}
            >
              <option value={UNMAPPED}>None (treat as single topic)</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <div style={{ fontSize: 11, color: PALETTE.textDim, marginTop: 4 }}>
              Groups utterances by topic for cross-topic analysis
            </div>
          </div>
        </div>

        {/* Live preview */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: PALETTE.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
            Preview — how your data will be interpreted
          </div>
          <div style={{ borderRadius: 8, border: `1px solid ${PALETTE.border}`, overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ background: PALETTE.bg }}>
                  <th style={{ padding: "8px 12px", textAlign: "left", color: PALETTE.textMuted, fontWeight: 600 }}>#</th>
                  <th style={{ padding: "8px 12px", textAlign: "left", color: utteranceCol ? PALETTE.accent : PALETTE.error, fontWeight: 600 }}>
                    {utteranceCol ? `utterance ← ${utteranceCol}` : "utterance ← ?"}
                  </th>
                  <th style={{ padding: "8px 12px", textAlign: "left", color: topicCol && topicCol !== UNMAPPED ? PALETTE.cyan : PALETTE.textDim, fontWeight: 600 }}>
                    {topicCol && topicCol !== UNMAPPED ? `topic_id ← ${topicCol}` : "topic_id ← (all \"unknown\")"}
                  </th>
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, i) => (
                  <tr key={i} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                    <td style={{ padding: "6px 12px", color: PALETTE.textDim }}>{i + 1}</td>
                    <td style={{ padding: "6px 12px", color: utteranceCol ? PALETTE.text : PALETTE.textDim }}>
                      {utteranceCol ? truncate(row[utteranceCol] || "", 80) : "—"}
                    </td>
                    <td style={{ padding: "6px 12px" }}>
                      {topicCol && topicCol !== UNMAPPED ? (
                        <Badge color={PALETTE.cyan}>{row[topicCol] || "unknown"}</Badge>
                      ) : (
                        <Badge color={PALETTE.textDim}>unknown</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confirm button */}
        <button
          onClick={handleConfirm}
          disabled={!isValid}
          style={{
            width: "100%",
            padding: "13px 24px",
            borderRadius: 10,
            border: "none",
            background: isValid
              ? `linear-gradient(135deg, ${PALETTE.accent}, ${PALETTE.purple})`
              : PALETTE.border,
            color: isValid ? "#fff" : PALETTE.textDim,
            fontSize: 14,
            fontWeight: 700,
            cursor: isValid ? "pointer" : "not-allowed",
            transition: "all 0.2s ease",
            boxShadow: isValid ? `0 4px 20px ${PALETTE.accentGlow}` : "none",
          }}
        >
          {isValid ? `✓ Confirm mapping & continue (${rows.length} rows)` : "Select an utterance column to continue"}
        </button>
      </Card>

      {/* All columns reference */}
      <Card style={{ background: PALETTE.bg, border: `1px dashed ${PALETTE.border}` }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: PALETTE.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          All columns in your file
        </div>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {columns.map((c) => {
            const isMappedUtterance = c === mapping.utterance;
            const isMappedTopic = c === mapping.topic_id;
            return (
              <Badge
                key={c}
                color={isMappedUtterance ? PALETTE.accent : isMappedTopic ? PALETTE.cyan : PALETTE.textDim}
              >
                {c}
                {isMappedUtterance && " → utterance"}
                {isMappedTopic && " → topic_id"}
              </Badge>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

// ─── Data Preview ───────────────────────────────────────────────────────────
function DataPreview({ data, filename }) {
  const topics = [...new Set(data.map((d) => d.topic_id))];
  return (
    <Card>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: PALETTE.text }}>📋 {filename}</div>
          <div style={{ fontSize: 12, color: PALETTE.textMuted, marginTop: 2 }}>
            {data.length} utterances · {topics.length} topic{topics.length !== 1 ? "s" : ""}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {topics.slice(0, 6).map((t) => (
            <Badge key={t} color={PALETTE.cyan}>{t}</Badge>
          ))}
          {topics.length > 6 && <Badge color={PALETTE.textMuted}>+{topics.length - 6} more</Badge>}
        </div>
      </div>
      <div style={{ maxHeight: 180, overflow: "auto", borderRadius: 8, border: `1px solid ${PALETTE.border}` }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: PALETTE.bg }}>
              <th style={{ padding: "8px 12px", textAlign: "left", color: PALETTE.textMuted, fontWeight: 600, position: "sticky", top: 0, background: PALETTE.bg }}>#</th>
              <th style={{ padding: "8px 12px", textAlign: "left", color: PALETTE.textMuted, fontWeight: 600, position: "sticky", top: 0, background: PALETTE.bg }}>Utterance</th>
              <th style={{ padding: "8px 12px", textAlign: "left", color: PALETTE.textMuted, fontWeight: 600, position: "sticky", top: 0, background: PALETTE.bg }}>Topic</th>
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 20).map((row, i) => (
              <tr key={i} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                <td style={{ padding: "6px 12px", color: PALETTE.textDim }}>{i + 1}</td>
                <td style={{ padding: "6px 12px", color: PALETTE.text }}>{truncate(row.utterance, 100)}</td>
                <td style={{ padding: "6px 12px" }}><Badge color={PALETTE.cyan}>{row.topic_id}</Badge></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > 20 && (
        <div style={{ textAlign: "center", padding: "8px 0 0", fontSize: 12, color: PALETTE.textDim }}>
          Showing 20 of {data.length} rows
        </div>
      )}
    </Card>
  );
}

// ─── Analysis Progress ──────────────────────────────────────────────────────
function AnalysisProgress({ stage, batchProgress }) {
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
        <div className="spinner" />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: PALETTE.text }}>Analyzing utterances…</div>
          <div style={{ fontSize: 12, color: PALETTE.textMuted, marginTop: 2 }}>{stage}</div>
        </div>
      </div>
      {batchProgress && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: PALETTE.textMuted }}>
              Batch {batchProgress.current} of {batchProgress.total}
            </span>
            <span style={{ fontSize: 12, color: PALETTE.accent, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
              {Math.round((batchProgress.current / batchProgress.total) * 100)}%
            </span>
          </div>
          <ProgressBar value={batchProgress.current} max={batchProgress.total} color={PALETTE.accent} height={8} />
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {(batchProgress?.steps || []).map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 20,
                height: 20,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 10,
                fontWeight: 700,
                background: s.done ? PALETTE.accent : s.active ? PALETTE.accentSoft : PALETTE.border,
                color: s.done ? "#fff" : s.active ? PALETTE.accent : PALETTE.textDim,
                transition: "all 0.3s ease",
              }}
            >
              {s.done ? "✓" : i + 1}
            </div>
            <span style={{ fontSize: 12, color: s.done || s.active ? PALETTE.text : PALETTE.textDim, fontWeight: s.active ? 600 : 400 }}>
              {s.label}
            </span>
          </div>
        ))}
      </div>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner {
          width: 20px; height: 20px;
          border: 2px solid ${PALETTE.border};
          border-top-color: ${PALETTE.accent};
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
      `}</style>
    </Card>
  );
}

// ─── Theme Card ─────────────────────────────────────────────────────────────
function ThemeCard({ theme, index, maxFreq, expanded, onToggle }) {
  const color = THEME_COLORS[index % THEME_COLORS.length];
  return (
    <Card hoverable onClick={onToggle} style={{ borderLeft: `3px solid ${color}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: PALETTE.text }}>{theme.name}</span>
            <SentimentBadge sentiment={theme.sentiment} />
            <SeverityBadge severity={theme.severity} />
          </div>
          <div style={{ fontSize: 13, color: PALETTE.textMuted, lineHeight: 1.5 }}>{theme.description}</div>
        </div>
        <div style={{ textAlign: "right", minWidth: 60 }}>
          <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>{theme.frequency}</div>
          <div style={{ fontSize: 10, color: PALETTE.textDim, textTransform: "uppercase", letterSpacing: "0.05em" }}>mentions</div>
        </div>
      </div>
      <div style={{ marginTop: 10 }}>
        <ProgressBar value={theme.frequency} max={maxFreq} color={color} />
      </div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 10 }}>
        {(theme.topic_ids || []).map((t) => (
          <Badge key={t} color={color}>{t}</Badge>
        ))}
      </div>
      {expanded && (
        <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px solid ${PALETTE.border}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: PALETTE.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
            Codes
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 14 }}>
            {(theme.codes || []).map((c, i) => (
              <Badge key={i} color={PALETTE.purple}>{c}</Badge>
            ))}
          </div>
          <div style={{ fontSize: 11, fontWeight: 600, color: PALETTE.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
            Example utterances
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {(theme.example_utterances || []).map((u, i) => (
              <div
                key={i}
                style={{
                  fontSize: 12,
                  color: PALETTE.text,
                  padding: "8px 12px",
                  borderRadius: 8,
                  background: PALETTE.bg,
                  borderLeft: `2px solid ${color}40`,
                  lineHeight: 1.5,
                }}
              >
                "{u}"
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ textAlign: "center", marginTop: 8 }}>
        <span style={{ fontSize: 11, color: PALETTE.textDim }}>{expanded ? "▲ Collapse" : "▼ Click to expand"}</span>
      </div>
    </Card>
  );
}

// ─── Export Utilities ────────────────────────────────────────────────────────
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportAsJSON(results, data) {
  const payload = {
    exported_at: new Date().toISOString(),
    source_rows: data.length,
    source_topics: [...new Set(data.map((d) => d.topic_id))],
    analysis: results,
  };
  downloadFile(JSON.stringify(payload, null, 2), `axial-coding-${Date.now()}.json`, "application/json");
}

function exportAsCSV(results) {
  const { themes = [] } = results;
  const headers = ["Theme", "Description", "Sentiment", "Severity", "Frequency", "Topics", "Codes", "Example Utterances"];
  const rows = themes.map((t) => [
    `"${(t.name || "").replace(/"/g, '""')}"`,
    `"${(t.description || "").replace(/"/g, '""')}"`,
    t.sentiment || "",
    t.severity || "",
    t.frequency || 0,
    `"${(t.topic_ids || []).join("; ")}"`,
    `"${(t.codes || []).join("; ")}"`,
    `"${(t.example_utterances || []).join(" | ").replace(/"/g, '""')}"`,
  ]);

  // Pain points sheet
  const ppHeaders = ["Issue", "Severity", "Affected Topics", "Recommendation"];
  const ppRows = (results.pain_points || []).map((p) => [
    `"${(p.issue || "").replace(/"/g, '""')}"`,
    p.severity || "",
    `"${(p.affected_topics || []).join("; ")}"`,
    `"${(p.recommendation || "").replace(/"/g, '""')}"`,
  ]);

  let csv = "THEMES\n" + headers.join(",") + "\n" + rows.map((r) => r.join(",")).join("\n");
  csv += "\n\nPAIN POINTS\n" + ppHeaders.join(",") + "\n" + ppRows.map((r) => r.join(",")).join("\n");

  // Summary
  const s = results.summary || {};
  csv += `\n\nSUMMARY\nCore Theme,"${(s.core_theme || "").replace(/"/g, '""')}"`;
  csv += `\nKey Insight,"${(s.key_insight || "").replace(/"/g, '""')}"`;
  csv += `\nOverall Sentiment,${s.overall_sentiment || ""}`;
  csv += `\nRisk Areas,"${(s.risk_areas || []).join("; ")}"`;
  csv += `\nOpportunities,"${(s.opportunity_areas || []).join("; ")}"`;

  downloadFile(csv, `axial-coding-${Date.now()}.csv`, "text/csv");
}

function exportAsMarkdown(results, data) {
  const { themes = [], cross_topic_patterns = [], pain_points = [], positive_signals = [], summary = {} } = results;
  const topics = [...new Set(data.map((d) => d.topic_id))];
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  let md = `# Axial Coding Analysis Report\n\n`;
  md += `**Generated:** ${date}  \n`;
  md += `**Utterances analyzed:** ${data.length}  \n`;
  md += `**Topics covered:** ${topics.join(", ")}  \n\n`;
  md += `---\n\n`;

  md += `## Executive Summary\n\n`;
  md += `**Core Theme:** ${summary.core_theme || "N/A"}\n\n`;
  md += `${summary.key_insight || ""}\n\n`;
  md += `**Overall Sentiment:** ${summary.overall_sentiment || "N/A"}\n\n`;

  if (summary.risk_areas?.length) {
    md += `### Risk Areas\n\n`;
    summary.risk_areas.forEach((r) => { md += `- ${r}\n`; });
    md += `\n`;
  }
  if (summary.opportunity_areas?.length) {
    md += `### Opportunities\n\n`;
    summary.opportunity_areas.forEach((o) => { md += `- ${o}\n`; });
    md += `\n`;
  }

  md += `---\n\n## Themes (${themes.length})\n\n`;
  themes.sort((a, b) => (b.frequency || 0) - (a.frequency || 0)).forEach((t, i) => {
    md += `### ${i + 1}. ${t.name}\n\n`;
    md += `| Sentiment | Severity | Frequency | Topics |\n`;
    md += `|-----------|----------|-----------|--------|\n`;
    md += `| ${t.sentiment} | ${t.severity} | ${t.frequency} | ${(t.topic_ids || []).join(", ")} |\n\n`;
    md += `${t.description}\n\n`;
    if (t.codes?.length) md += `**Codes:** ${t.codes.join(", ")}\n\n`;
    if (t.example_utterances?.length) {
      md += `**Examples:**\n\n`;
      t.example_utterances.forEach((u) => { md += `> "${u}"\n\n`; });
    }
  });

  if (pain_points.length) {
    md += `---\n\n## Pain Points (${pain_points.length})\n\n`;
    pain_points.forEach((p) => {
      md += `- **[${(p.severity || "medium").toUpperCase()}]** ${p.issue}`;
      if (p.recommendation) md += ` → *${p.recommendation}*`;
      md += `\n`;
    });
    md += `\n`;
  }

  if (cross_topic_patterns.length) {
    md += `---\n\n## Cross-Topic Patterns (${cross_topic_patterns.length})\n\n`;
    cross_topic_patterns.forEach((p) => {
      md += `- **${p.pattern}** (${(p.topics_affected || []).join(", ")}): ${p.implication}\n`;
    });
    md += `\n`;
  }

  if (positive_signals.length) {
    md += `---\n\n## Positive Signals (${positive_signals.length})\n\n`;
    positive_signals.forEach((p) => {
      md += `- **[${p.strength || "moderate"}]** ${p.signal} (${(p.topics || []).join(", ")})\n`;
    });
  }

  downloadFile(md, `axial-coding-report-${Date.now()}.md`, "text/markdown");
}

// ─── Export Toolbar ─────────────────────────────────────────────────────────
function ExportToolbar({ results, data }) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setShowMenu(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const exports = [
    { label: "Markdown Report", sub: "Formatted report with all sections", icon: "📝", fn: () => exportAsMarkdown(results, data) },
    { label: "CSV Spreadsheet", sub: "Themes + pain points in tabular format", icon: "📊", fn: () => exportAsCSV(results) },
    { label: "Raw JSON", sub: "Full structured data for pipelines", icon: "{ }", fn: () => exportAsJSON(results, data) },
  ];

  return (
    <div ref={menuRef} style={{ position: "relative" }}>
      <button
        onClick={() => setShowMenu(!showMenu)}
        style={{
          padding: "9px 18px",
          borderRadius: 8,
          border: `1px solid ${PALETTE.border}`,
          background: showMenu ? PALETTE.surfaceHover : PALETTE.surface,
          color: PALETTE.text,
          fontSize: 13,
          fontWeight: 600,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 7,
          transition: "all 0.15s ease",
        }}
      >
        <span style={{ fontSize: 14 }}>↓</span>
        Export
        <span style={{ fontSize: 10, color: PALETTE.textDim, transition: "transform 0.2s", transform: showMenu ? "rotate(180deg)" : "rotate(0)" }}>▼</span>
      </button>
      {showMenu && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            right: 0,
            width: 260,
            background: PALETTE.surface,
            border: `1px solid ${PALETTE.border}`,
            borderRadius: 10,
            padding: 6,
            zIndex: 100,
            boxShadow: `0 8px 30px rgba(0,0,0,0.4)`,
          }}
        >
          {exports.map((exp, i) => (
            <button
              key={i}
              onClick={() => { exp.fn(); setShowMenu(false); }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                width: "100%",
                padding: "10px 12px",
                borderRadius: 7,
                border: "none",
                background: "transparent",
                cursor: "pointer",
                textAlign: "left",
                transition: "background 0.12s ease",
              }}
              onMouseOver={(e) => (e.currentTarget.style.background = PALETTE.surfaceHover)}
              onMouseOut={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 7,
                  background: PALETTE.accentSoft,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  flexShrink: 0,
                }}
              >
                {exp.icon}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: PALETTE.text }}>{exp.label}</div>
                <div style={{ fontSize: 11, color: PALETTE.textDim, marginTop: 1 }}>{exp.sub}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Results Dashboard ──────────────────────────────────────────────────────
function ResultsDashboard({ results, data }) {
  const [expandedTheme, setExpandedTheme] = useState(null);
  const [activeTab, setActiveTab] = useState("themes");

  const { themes = [], cross_topic_patterns = [], pain_points = [], positive_signals = [], summary = {} } = results;
  const maxFreq = Math.max(...themes.map((t) => t.frequency || 0), 1);
  const topics = [...new Set(data.map((d) => d.topic_id))];

  const tabs = [
    { id: "themes", label: "Themes", icon: "🏷️", count: themes.length },
    { id: "patterns", label: "Cross-topic", icon: "🔗", count: cross_topic_patterns.length },
    { id: "pain", label: "Pain Points", icon: "⚠️", count: pain_points.length },
    { id: "positive", label: "Positive Signals", icon: "✅", count: positive_signals.length },
  ];

  // Sentiment distribution
  const sentimentCounts = { positive: 0, negative: 0, neutral: 0, mixed: 0 };
  themes.forEach((t) => { sentimentCounts[t.sentiment] = (sentimentCounts[t.sentiment] || 0) + (t.frequency || 1); });
  const totalSent = Object.values(sentimentCounts).reduce((a, b) => a + b, 0) || 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Summary Banner */}
      <Card style={{ background: `linear-gradient(135deg, ${PALETTE.surface} 0%, #0f1729 100%)`, border: `1px solid ${PALETTE.accent}30` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: PALETTE.accent, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
              Core Theme
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: PALETTE.text, lineHeight: 1.4, marginBottom: 10 }}>
              {summary.core_theme || "Analysis complete"}
            </div>
            <div style={{ fontSize: 13, color: PALETTE.textMuted, lineHeight: 1.6 }}>
              {summary.key_insight}
            </div>
          </div>
          <ExportToolbar results={results} data={data} />
        </div>
      </Card>

      {/* Stat Cards */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <StatCard label="Utterances" value={summary.total_utterances_analyzed || data.length} icon="💬" color={PALETTE.accent} />
        <StatCard label="Topics" value={summary.total_topics || topics.length} icon="📁" color={PALETTE.cyan} />
        <StatCard label="Themes Found" value={themes.length} icon="🏷️" color={PALETTE.purple} />
        <StatCard label="Pain Points" value={pain_points.length} icon="⚠️" color={PALETTE.error} />
      </div>

      {/* Sentiment Distribution */}
      <Card>
        <div style={{ fontSize: 13, fontWeight: 600, color: PALETTE.text, marginBottom: 12 }}>Sentiment Distribution</div>
        <div style={{ display: "flex", height: 10, borderRadius: 5, overflow: "hidden", gap: 2 }}>
          {sentimentCounts.positive > 0 && <div style={{ flex: sentimentCounts.positive, background: PALETTE.success, borderRadius: 5 }} title={`Positive: ${sentimentCounts.positive}`} />}
          {sentimentCounts.neutral > 0 && <div style={{ flex: sentimentCounts.neutral, background: PALETTE.textDim, borderRadius: 5 }} title={`Neutral: ${sentimentCounts.neutral}`} />}
          {sentimentCounts.mixed > 0 && <div style={{ flex: sentimentCounts.mixed, background: PALETTE.warning, borderRadius: 5 }} title={`Mixed: ${sentimentCounts.mixed}`} />}
          {sentimentCounts.negative > 0 && <div style={{ flex: sentimentCounts.negative, background: PALETTE.error, borderRadius: 5 }} title={`Negative: ${sentimentCounts.negative}`} />}
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
          {Object.entries(sentimentCounts).filter(([, v]) => v > 0).map(([k, v]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: k === "positive" ? PALETTE.success : k === "negative" ? PALETTE.error : k === "mixed" ? PALETTE.warning : PALETTE.textDim }} />
              <span style={{ color: PALETTE.textMuted, textTransform: "capitalize" }}>{k}</span>
              <span style={{ color: PALETTE.text, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{Math.round((v / totalSent) * 100)}%</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Risk & Opportunities */}
      {(summary.risk_areas?.length > 0 || summary.opportunity_areas?.length > 0) && (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {summary.risk_areas?.length > 0 && (
            <Card style={{ flex: "1 1 280px" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: PALETTE.error, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                🔴 Risk Areas
              </div>
              {summary.risk_areas.map((r, i) => (
                <div key={i} style={{ fontSize: 13, color: PALETTE.text, padding: "8px 12px", borderRadius: 8, background: PALETTE.errorSoft, marginBottom: 6, lineHeight: 1.5 }}>{r}</div>
              ))}
            </Card>
          )}
          {summary.opportunity_areas?.length > 0 && (
            <Card style={{ flex: "1 1 280px" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: PALETTE.success, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                🟢 Opportunities
              </div>
              {summary.opportunity_areas.map((o, i) => (
                <div key={i} style={{ fontSize: 13, color: PALETTE.text, padding: "8px 12px", borderRadius: 8, background: PALETTE.successSoft, marginBottom: 6, lineHeight: 1.5 }}>{o}</div>
              ))}
            </Card>
          )}
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, background: PALETTE.bg, padding: 4, borderRadius: 10, border: `1px solid ${PALETTE.border}` }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 8,
              border: "none",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              background: activeTab === tab.id ? PALETTE.surface : "transparent",
              color: activeTab === tab.id ? PALETTE.text : PALETTE.textDim,
              transition: "all 0.2s ease",
            }}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
            <span
              style={{
                background: activeTab === tab.id ? PALETTE.accentSoft : PALETTE.border,
                color: activeTab === tab.id ? PALETTE.accent : PALETTE.textDim,
                padding: "1px 7px",
                borderRadius: 99,
                fontSize: 11,
                fontWeight: 700,
              }}
            >
              {tab.count}
            </span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "themes" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {themes.sort((a, b) => (b.frequency || 0) - (a.frequency || 0)).map((theme, i) => (
            <ThemeCard
              key={i}
              theme={theme}
              index={i}
              maxFreq={maxFreq}
              expanded={expandedTheme === i}
              onToggle={() => setExpandedTheme(expandedTheme === i ? null : i)}
            />
          ))}
        </div>
      )}

      {activeTab === "patterns" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {cross_topic_patterns.length === 0 ? (
            <Card><div style={{ textAlign: "center", color: PALETTE.textDim, padding: 20 }}>No cross-topic patterns detected</div></Card>
          ) : (
            cross_topic_patterns.map((p, i) => (
              <Card key={i}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <span style={{ fontSize: 14 }}>🔗</span>
                  <span style={{ fontSize: 14, fontWeight: 600, color: PALETTE.text }}>{p.pattern}</span>
                </div>
                <div style={{ fontSize: 13, color: PALETTE.textMuted, lineHeight: 1.5, marginBottom: 10 }}>{p.implication}</div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {(p.topics_affected || []).map((t) => (
                    <Badge key={t} color={PALETTE.cyan}>{t}</Badge>
                  ))}
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {activeTab === "pain" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {pain_points.length === 0 ? (
            <Card><div style={{ textAlign: "center", color: PALETTE.textDim, padding: 20 }}>No pain points detected — looking good!</div></Card>
          ) : (
            pain_points.map((p, i) => (
              <Card key={i} style={{ borderLeft: `3px solid ${p.severity === "high" ? PALETTE.error : p.severity === "medium" ? PALETTE.warning : PALETTE.textDim}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <SeverityBadge severity={p.severity} />
                  <span style={{ fontSize: 14, fontWeight: 600, color: PALETTE.text }}>{p.issue}</span>
                </div>
                {p.recommendation && (
                  <div style={{ fontSize: 13, color: PALETTE.success, padding: "8px 12px", borderRadius: 8, background: PALETTE.successSoft, marginTop: 8 }}>
                    💡 {p.recommendation}
                  </div>
                )}
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 8 }}>
                  {(p.affected_topics || []).map((t) => (
                    <Badge key={t} color={PALETTE.error}>{t}</Badge>
                  ))}
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {activeTab === "positive" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {positive_signals.length === 0 ? (
            <Card><div style={{ textAlign: "center", color: PALETTE.textDim, padding: 20 }}>No positive signals detected</div></Card>
          ) : (
            positive_signals.map((p, i) => (
              <Card key={i} style={{ borderLeft: `3px solid ${PALETTE.success}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <Badge color={PALETTE.success}>{p.strength || "moderate"}</Badge>
                  <span style={{ fontSize: 14, fontWeight: 600, color: PALETTE.text }}>{p.signal}</span>
                </div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 8 }}>
                  {(p.topics || []).map((t) => (
                    <Badge key={t} color={PALETTE.success}>{t}</Badge>
                  ))}
                </div>
              </Card>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [rawFile, setRawFile] = useState(null); // { columns, rows, filename }
  const [data, setData] = useState(null);
  const [filename, setFilename] = useState("");
  const [results, setResults] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [stage, setStage] = useState("");
  const [batchProgress, setBatchProgress] = useState(null);
  const [error, setError] = useState(null);

  const handleFileParsed = useCallback((fileData) => {
    setRawFile(fileData);
    setData(null);
    setResults(null);
    setError(null);
  }, []);

  const handleMappingConfirmed = useCallback((parsed, fname) => {
    setData(parsed);
    setFilename(fname);
    setResults(null);
    setError(null);
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!data || data.length === 0) return;
    setAnalyzing(true);
    setError(null);
    setResults(null);

    const topics = [...new Set(data.map((d) => d.topic_id))];

    try {
      const batches = chunkUtterances(data);
      const totalBatches = batches.length;
      const needsSynthesis = totalBatches > 1;

      const makeSteps = (currentStep) => {
        const steps = [
          { label: `Open coding${needsSynthesis ? ` (${totalBatches} batches)` : ""}`, done: currentStep > 0, active: currentStep === 0 },
        ];
        if (needsSynthesis) {
          steps.push({ label: "Axial coding — merging & synthesizing", done: currentStep > 1, active: currentStep === 1 });
        }
        steps.push({ label: "Building insights dashboard", done: currentStep > (needsSynthesis ? 2 : 1), active: currentStep === (needsSynthesis ? 2 : 1) });
        return steps;
      };

      // ── Phase 1: Open coding (batched) ──
      const batchResults = [];

      for (let i = 0; i < batches.length; i++) {
        setStage(`Open coding batch ${i + 1} of ${totalBatches}…`);
        setBatchProgress({ current: i, total: totalBatches, steps: makeSteps(0) });

        const prompt = buildBatchCodingPrompt(batches[i]);
        const result = await callClaude(prompt);
        batchResults.push(result);

        setBatchProgress({ current: i + 1, total: totalBatches, steps: makeSteps(0) });
      }

      // ── Phase 2: Synthesis (if multiple batches) ──
      let finalResult;
      if (needsSynthesis) {
        setStage("Synthesizing themes across batches…");
        setBatchProgress({ current: totalBatches, total: totalBatches, steps: makeSteps(1) });

        const synthesisPrompt = buildSynthesisPrompt(batchResults, data.length, topics.length);
        finalResult = await callClaude(synthesisPrompt);
      } else {
        finalResult = batchResults[0];
      }

      // ── Phase 3: Finalize ──
      setStage("Building insights dashboard…");
      setBatchProgress({ current: totalBatches, total: totalBatches, steps: makeSteps(needsSynthesis ? 2 : 1) });
      await new Promise((r) => setTimeout(r, 400));

      // Ensure summary counts are correct
      if (finalResult.summary) {
        finalResult.summary.total_utterances_analyzed = data.length;
        finalResult.summary.total_topics = topics.length;
      }

      setResults(finalResult);
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setAnalyzing(false);
      setStage("");
      setBatchProgress(null);
    }
  }, [data]);

  const reset = () => {
    setRawFile(null);
    setData(null);
    setFilename("");
    setResults(null);
    setError(null);
    setAnalyzing(false);
    setStage("");
    setBatchProgress(null);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: PALETTE.bg,
        color: PALETTE.text,
        fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div
        style={{
          padding: "20px 32px",
          borderBottom: `1px solid ${PALETTE.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: `linear-gradient(180deg, ${PALETTE.surface} 0%, ${PALETTE.bg} 100%)`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              background: "#C8102E",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              fontWeight: 700,
              color: "#fff",
            }}
          >
            ✓
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: PALETTE.text, letterSpacing: "-0.01em" }}>
              Turbox Tax Labs
            </div>
            <div style={{ fontSize: 11, color: PALETTE.textMuted }}>Utterance Theme Analyzer for PMs</div>
          </div>
        </div>
        {(data || rawFile) && (
          <button
            onClick={reset}
            style={{
              padding: "7px 16px",
              borderRadius: 8,
              border: `1px solid ${PALETTE.border}`,
              background: "transparent",
              color: PALETTE.textMuted,
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            ↺ New Analysis
          </button>
        )}
      </div>

      {/* Content */}
      <div style={{ maxWidth: 800, margin: "0 auto", padding: "28px 24px 60px" }}>
        {!rawFile && !data ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <div style={{ textAlign: "center", paddingTop: 20 }}>
              <h1 style={{ fontSize: 28, fontWeight: 700, color: PALETTE.text, margin: "0 0 8px", letterSpacing: "-0.02em" }}>
                Analyze your agent utterances
              </h1>
              <p style={{ fontSize: 15, color: PALETTE.textMuted, margin: 0, maxWidth: 520, marginLeft: "auto", marginRight: "auto", lineHeight: 1.6 }}>
                Upload a CSV of customer utterances from your agent UX. Get axial-coded thematic analysis with pain points, cross-topic patterns, and actionable insights in seconds, not days.
              </p>
            </div>
            <UploadZone onFileParsed={handleFileParsed} />
            <Card style={{ background: PALETTE.bg, border: `1px dashed ${PALETTE.border}` }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: PALETTE.textMuted, marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                How it works
              </div>
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                {[
                  { n: "1", t: "Upload", d: "Drop any CSV — no fixed column names required" },
                  { n: "2", t: "Map", d: "Map your columns to utterance & topic (auto-detected)" },
                  { n: "3", t: "Analyze", d: "We leverage LLMs plus our custom axial doing rules to perform open → axial → selective coding" },
                ].map((s) => (
                  <div key={s.n} style={{ flex: "1 1 160px", display: "flex", gap: 10, alignItems: "flex-start" }}>
                    <div
                      style={{
                        width: 24,
                        height: 24,
                        borderRadius: 6,
                        background: PALETTE.accentSoft,
                        color: PALETTE.accent,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 12,
                        fontWeight: 700,
                        flexShrink: 0,
                      }}
                    >
                      {s.n}
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: PALETTE.text }}>{s.t}</div>
                      <div style={{ fontSize: 12, color: PALETTE.textDim, lineHeight: 1.5 }}>{s.d}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        ) : rawFile && !data ? (
          <ColumnMapper
            columns={rawFile.columns}
            rows={rawFile.rows}
            filename={rawFile.filename}
            onConfirm={handleMappingConfirmed}
            onBack={() => setRawFile(null)}
          />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <DataPreview data={data} filename={filename} />

            {!analyzing && !results && (
              <button
                onClick={runAnalysis}
                style={{
                  padding: "14px 28px",
                  borderRadius: 10,
                  border: "none",
                  background: `linear-gradient(135deg, ${PALETTE.accent}, ${PALETTE.purple})`,
                  color: "#fff",
                  fontSize: 15,
                  fontWeight: 700,
                  cursor: "pointer",
                  letterSpacing: "0.01em",
                  transition: "opacity 0.2s ease",
                  boxShadow: `0 4px 20px ${PALETTE.accentGlow}`,
                }}
                onMouseOver={(e) => (e.target.style.opacity = "0.9")}
                onMouseOut={(e) => (e.target.style.opacity = "1")}
              >
                ▶ Run Axial Coding Analysis
              </button>
            )}

            {analyzing && <AnalysisProgress stage={stage} batchProgress={batchProgress} />}

            {error && (
              <Card style={{ borderColor: PALETTE.error }}>
                <div style={{ color: PALETTE.error, fontSize: 14, fontWeight: 600, marginBottom: 6 }}>Analysis failed</div>
                <div style={{ color: PALETTE.textMuted, fontSize: 13, lineHeight: 1.5 }}>{error}</div>
                <button
                  onClick={runAnalysis}
                  style={{
                    marginTop: 12,
                    padding: "8px 20px",
                    borderRadius: 8,
                    border: `1px solid ${PALETTE.error}`,
                    background: PALETTE.errorSoft,
                    color: PALETTE.error,
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Retry
                </button>
              </Card>
            )}

            {results && <ResultsDashboard results={results} data={data} />}
          </div>
        )}
      </div>
    </div>
  );
}
