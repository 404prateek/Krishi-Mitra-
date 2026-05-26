import { useState, useEffect, useCallback } from "react";
import { parseAdvisoryText } from "./DiseaseResultCard";
import DiseaseResultCard from "./DiseaseResultCard";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const SEV_COLOR = { high: "#ef4444", medium: "#f59e0b", low: "#22c55e" };
const SEV_BG    = { high: "#fef2f2", medium: "#fffbeb", low: "#f0fdf4" };

export default function ScanHistory({ language = "en" }) {
  const [scans, setScans]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [selected, setSelected] = useState(null); // full result to show in card
  const [deleting, setDeleting] = useState(null);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/scan-history?limit=30`);
      if (!r.ok) return;
      const { scans: data } = await r.json();
      setScans(data || []);
    } catch { /* backend may be offline */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const openScan = (scan) => {
    const text = scan.advisory_detail || scan.advisory_short || "";
    setSelected({
      scan,
      parsed: {
        diseaseName: scan.disease,
        crop:        scan.crop,
        confidence:  scan.confidence,
        severity:    scan.severity,
        isHealthy:   scan.is_healthy === 1,
        sections:    parseAdvisoryText(text),
        rawText:     text || "No advisory available.",
        economicImpact: scan.economic_impact || "",
      },
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleDelete = async (e, scanId) => {
    e.stopPropagation();
    if (!window.confirm("Remove this scan from history?")) return;
    setDeleting(scanId);
    try {
      await fetch(`${API}/api/scan-history/${scanId}`, { method: "DELETE" });
      setScans(prev => prev.filter(s => s.id !== scanId));
      if (selected?.scan?.id === scanId) setSelected(null);
    } finally { setDeleting(null); }
  };

  if (loading) return null;
  if (scans.length === 0) return null;

  return (
    <section style={S.section}>
      <div style={S.header}>
        <h2 style={S.heading}>📋 Recent Scans</h2>
        <span style={S.count}>{scans.length} scan{scans.length !== 1 ? "s" : ""}</span>
      </div>

      {/* ── Full result card overlay ── */}
      {selected && (
        <div style={S.overlay}>
          <div style={S.overlayInner}>
            <div style={S.overlayTop}>
              <h3 style={S.overlayTitle}>Scan Result</h3>
              <button style={S.closeBtn} onClick={() => setSelected(null)}>✕ Close</button>
            </div>
            <div style={S.overlayImg}>
              <img
                src={`${API}/api/scan-history/${selected.scan.id}/image`}
                alt="Scanned crop"
                style={S.bigImg}
                onError={e => { e.target.style.display = "none"; }}
              />
            </div>
            <DiseaseResultCard
              parsed={selected.parsed}
              language={language}
              economicImpact={selected.scan.economic_impact}
              onScanAgain={() => setSelected(null)}
            />
          </div>
        </div>
      )}

      {/* ── Horizontal scroll grid ── */}
      <div style={S.grid}>
        {scans.map(scan => {
          const color = SEV_COLOR[scan.severity] || SEV_COLOR.medium;
          const bg    = SEV_BG[scan.severity]    || SEV_BG.medium;
          const date  = new Date(scan.created_at).toLocaleDateString("en-IN", {
            day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
          });
          return (
            <div
              key={scan.id}
              style={{ ...S.card, borderColor: color }}
              onClick={() => openScan(scan)}
              title="Click to view full report"
            >
              {/* Thumbnail */}
              <div style={{ ...S.imgWrap, background: bg }}>
                <img
                  src={`${API}/api/scan-history/${scan.id}/image`}
                  alt={scan.disease}
                  style={S.thumb}
                  onError={e => {
                    e.target.style.display = "none";
                    e.target.nextSibling.style.display = "flex";
                  }}
                />
                <div style={{ ...S.imgFallback, display: "none" }}>🌿</div>

                {/* Severity badge */}
                <span style={{ ...S.badge, background: color }}>
                  {scan.is_healthy ? "Healthy" : scan.severity.toUpperCase()}
                </span>

                {/* Delete button */}
                <button
                  style={S.deleteBtn}
                  onClick={e => handleDelete(e, scan.id)}
                  title="Delete"
                  disabled={deleting === scan.id}
                >
                  {deleting === scan.id ? "…" : "✕"}
                </button>
              </div>

              {/* Info */}
              <div style={S.info}>
                <div style={S.diseaseName}>
                  {scan.is_healthy ? `✅ ${scan.crop}` : scan.disease}
                </div>
                <div style={S.cropName}>{scan.crop}</div>
                <div style={S.meta}>{scan.confidence.toFixed(1)}% • {date}</div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

const S = {
  section: {
    background: "#f8fafc",
    borderTop: "1px solid #e5e7eb",
    padding: "40px 24px 48px",
    fontFamily: "'Inter', system-ui, sans-serif",
  },
  header: {
    maxWidth: 1100,
    margin: "0 auto 24px",
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  heading: {
    fontSize: 22,
    fontWeight: 700,
    color: "#111827",
    margin: 0,
  },
  count: {
    background: "#e0f2fe",
    color: "#0369a1",
    borderRadius: 20,
    padding: "2px 10px",
    fontSize: 13,
    fontWeight: 600,
  },
  grid: {
    maxWidth: 1100,
    margin: "0 auto",
    display: "flex",
    gap: 16,
    overflowX: "auto",
    paddingBottom: 8,
    scrollSnapType: "x mandatory",
  },
  card: {
    flex: "0 0 175px",
    background: "#fff",
    borderRadius: 14,
    border: "2px solid #e5e7eb",
    overflow: "hidden",
    cursor: "pointer",
    scrollSnapAlign: "start",
    transition: "transform 0.18s, box-shadow 0.18s",
    boxShadow: "0 1px 6px rgba(0,0,0,0.06)",
  },
  imgWrap: {
    position: "relative",
    width: "100%",
    paddingBottom: "75%",
    overflow: "hidden",
  },
  thumb: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  imgFallback: {
    position: "absolute",
    inset: 0,
    alignItems: "center",
    justifyContent: "center",
    fontSize: 36,
  },
  badge: {
    position: "absolute",
    top: 8,
    left: 8,
    color: "#fff",
    fontSize: 10,
    fontWeight: 700,
    padding: "2px 7px",
    borderRadius: 6,
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  deleteBtn: {
    position: "absolute",
    top: 6,
    right: 6,
    background: "rgba(0,0,0,0.45)",
    border: "none",
    color: "#fff",
    borderRadius: "50%",
    width: 22,
    height: 22,
    fontSize: 11,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    lineHeight: 1,
  },
  info: {
    padding: "10px 12px 12px",
  },
  diseaseName: {
    fontSize: 13,
    fontWeight: 700,
    color: "#111827",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  cropName: {
    fontSize: 12,
    color: "#6b7280",
    marginTop: 2,
  },
  meta: {
    fontSize: 11,
    color: "#9ca3af",
    marginTop: 4,
  },
  // overlay
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.6)",
    zIndex: 9999,
    overflowY: "auto",
    display: "flex",
    justifyContent: "center",
    padding: "24px 16px",
  },
  overlayInner: {
    background: "#fff",
    borderRadius: 20,
    maxWidth: 560,
    width: "100%",
    padding: 24,
    height: "fit-content",
  },
  overlayTop: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  overlayTitle: {
    margin: 0,
    fontSize: 18,
    fontWeight: 700,
    color: "#111827",
  },
  closeBtn: {
    background: "#f3f4f6",
    border: "none",
    borderRadius: 8,
    padding: "6px 14px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 13,
    color: "#374151",
  },
  overlayImg: {
    marginBottom: 16,
    borderRadius: 12,
    overflow: "hidden",
    maxHeight: 220,
    display: "flex",
    justifyContent: "center",
    background: "#f3f4f6",
  },
  bigImg: {
    maxWidth: "100%",
    maxHeight: 220,
    objectFit: "contain",
    display: "block",
  },
};
