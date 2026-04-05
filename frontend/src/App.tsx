import React, { useState, useRef, useCallback } from 'react';
import { UploadCloud, Sparkles, AlertCircle, CheckCircle2, BookOpen, Loader2, ChevronLeft, ChevronRight, X, ChevronDown, ChevronUp } from 'lucide-react';
import './App.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface QAItem {
  number: number;
  section: string;
  question: string;
  answer: string;
  rubric: { max_marks: number; reference_long: string; reference_short: string[]; keywords: { term: string; weight: number }[] };
}

interface RubricType {
  max_marks?: number;
  reference_long?: string;
  reference_short?: string[];
  keywords?: { term: string; weight: number }[];
}

interface AnswerResult {
  q_label: string;
  y_pct: number;
  auto_matched_question?: number;
  matched_question_text?: string;
  rubric?: RubricType;
  student_text?: string;
  full_page_text?: string;
  component_scores?: { keyword_pct?: number; grammar_pct?: number; semantic_pct?: number; coverage_pct?: number; presentation_pct?: number };
  llm_output?: { recommended_marks?: number; explanation?: string };
  deterministic_recommended_marks?: number;
  composite_confidence?: number;
}

interface PageEntry {
  pageIndex: number;
  blob: Blob;
  url: string;
  results: AnswerResult[];
  loading: boolean;
  error: string | null;
}

const API = 'http://localhost:8000';

function Bar({ label, val }: { label: string; val: number }) {
  return (
    <div className="metric">
      <div className="metric-row"><span>{label}</span><span>{Math.round(val)}%</span></div>
      <div className="bar-bg"><div className="bar-fill" style={{ width: `${val}%` }} /></div>
    </div>
  );
}

function AnswerPopup({
  res, onClose, top, onSelect, qaDataset, manualQNum, onManualSelect
}: {
  res: AnswerResult; onClose: () => void; top: string; onSelect: () => void;
  qaDataset: QAItem[]; manualQNum: number | null; onManualSelect: (n: number | null) => void;
}) {
  const lm = res.llm_output?.recommended_marks ?? 0;
  const dm = res.deterministic_recommended_marks ?? 0;
  const conf = res.composite_confidence ?? 0;
  const finalScore = (conf > 0.6 && lm > 0) ? lm : dm;
  const sc = res.component_scores ?? {};
  const maxM = res.rubric?.max_marks ?? 10;
  const displayQ = manualQNum ?? res.auto_matched_question;

  return (
    <div className="answer-popup" style={{ top }} onClick={onSelect}>
      <button className="popup-close" onClick={e => { e.stopPropagation(); onClose(); }}><X size={16} /></button>

      {/* Header: chip + question selector */}
      <div style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {manualQNum != null
          ? <span className="chip chip-yellow">✏️ Q{manualQNum}</span>
          : displayQ != null
            ? <span className="chip chip-green">🎯 Q{displayQ}</span>
            : <span className="chip chip-yellow">⚙️ Q{res.q_label}</span>}
        <span style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>/ {maxM} marks</span>
      </div>

      {/* Manual question selector (only when qa_dataset loaded) */}
      {qaDataset.length > 0 && (
        <div style={{ marginBottom: '0.55rem' }} onClick={e => e.stopPropagation()}>
          <select
            className="popup-q-select"
            value={manualQNum ?? ''}
            onChange={e => onManualSelect(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">🤖 Auto ({displayQ != null ? `Q${displayQ}` : 'unmatched'})</option>
            {qaDataset.map(q => (
              <option key={q.number} value={q.number}>Q{q.number} — {q.question.slice(0, 50)}{q.question.length > 50 ? '…' : ''}</option>
            ))}
          </select>
        </div>
      )}

      {/* Score Big Display */}
      <div style={{ background: 'rgba(16,185,129,.1)', border: '1px solid var(--success)', borderRadius: 8, padding: '0.5rem', textAlign: 'center', marginBottom: '0.6rem' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Score</div>
        <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--success)', lineHeight: 1.1 }}>{finalScore.toFixed(1)}</div>
        <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>of {maxM}</div>
      </div>

      <Bar label="Keywords" val={sc.keyword_pct ?? 0} />
      <Bar label="Semantic" val={sc.semantic_pct ?? 0} />
      <Bar label="Grammar" val={sc.grammar_pct ?? 0} />
      <Bar label="Presentation" val={sc.presentation_pct ?? 0} />

      {/* Extracted Text */}
      {res.student_text && (
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 2 }}>📝 Extracted Text</div>
          <div className="popup-extracted">{res.student_text}</div>
        </div>
      )}

      {res.llm_output?.explanation && (
        <div className="feedback-box" style={{ marginTop: '0.5rem' }}>
          <strong style={{ fontSize: '0.78rem' }}>🤖 Feedback</strong>
          <div style={{ marginTop: '0.3rem', fontSize: '0.78rem' }}>{res.llm_output.explanation}</div>
        </div>
      )}

      <div style={{ marginTop: '0.4rem', fontSize: '0.68rem', color: 'var(--muted)', textAlign: 'right' }}>
        Click panel to view rubric →
      </div>
    </div>
  );
}

// ─── RubricPanel ──────────────────────────────────────────────────────────────

function RubricPanel({ activeResult, qaDataset, manualOverrides, setManualOverrides, onBack }: {
  activeResult: AnswerResult;
  qaDataset: QAItem[];
  manualOverrides: Record<string, number | null>;
  setManualOverrides: React.Dispatch<React.SetStateAction<Record<string, number | null>>>;
  onBack: () => void;
}) {
  const manualNum = manualOverrides[activeResult.q_label] ?? null;
  const overrideItem = manualNum != null ? qaDataset.find(q => q.number === manualNum) : null;
  const displayRubric = overrideItem ? overrideItem.rubric : activeResult.rubric;
  const displayQ = overrideItem ? overrideItem.number : activeResult.auto_matched_question;
  const displayQText = overrideItem ? overrideItem.question : activeResult.matched_question_text;

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.5rem', flexWrap: 'wrap' }}>
        {overrideItem
          ? <span className="chip chip-yellow">✏️ Q{displayQ} (manual)</span>
          : displayQ != null
            ? <span className="chip chip-green">🎯 Q{displayQ}</span>
            : <span className="chip chip-yellow">⚙️ Q{activeResult.q_label}</span>}
        <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
          Max {displayRubric?.max_marks ?? 10} marks
        </span>
      </div>

      {qaDataset.length > 0 && (
        <div style={{ marginBottom: '0.6rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 3 }}>Assign to Question</div>
          <select
            className="popup-q-select"
            style={{ width: '100%' }}
            value={manualNum ?? ''}
            onChange={e => setManualOverrides(prev => ({ ...prev, [activeResult.q_label]: e.target.value ? Number(e.target.value) : null }))}
          >
            <option value="">🤖 Auto ({displayQ != null ? `Q${displayQ}` : 'unmatched'})</option>
            {qaDataset.map(q => (
              <option key={q.number} value={q.number}>Q{q.number} — {q.question.slice(0, 55)}{q.question.length > 55 ? '…' : ''}</option>
            ))}
          </select>
        </div>
      )}

      {displayQText && (
        <div style={{ marginBottom: '0.6rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 2 }}>Question</div>
          <div style={{ fontSize: '0.82rem', lineHeight: 1.4, color: 'var(--text)', background: 'rgba(255,255,255,0.04)', borderRadius: 6, padding: '0.5rem' }}>
            {displayQText}
          </div>
        </div>
      )}

      {displayRubric?.reference_long && (
        <div style={{ marginBottom: '0.6rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 2 }}>Expected Answer</div>
          <div className="ocr-box" style={{ maxHeight: 80 }}>{displayRubric.reference_long}</div>
        </div>
      )}

      {displayRubric?.keywords && displayRubric.keywords.length > 0 && (
        <div style={{ marginBottom: '0.5rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 4 }}>Keywords</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {displayRubric.keywords.map((kw, i) => (
              <span key={i} className="chip chip-green" style={{ fontSize: '0.7rem' }}>{kw.term}</span>
            ))}
          </div>
        </div>
      )}

      <button className="btn-ghost btn-sm" style={{ marginTop: '0.4rem', width: '100%' }}
        onClick={onBack}>← Back to manual form</button>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [qaDataset, setQaDataset] = useState<QAItem[]>([]);
  const [loadingQA, setLoadingQA] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);
  const qPdfRef = useRef<HTMLInputElement>(null);
  const aPdfRef = useRef<HTMLInputElement>(null);

  const [refLong, setRefLong] = useState("Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.");
  const [refShort, setRefShort] = useState('rest, motion, unbalanced force');
  const [keywords, setKeywords] = useState('inertia, rest, motion, force');
  const [maxMarks, setMaxMarks] = useState(10);

  const [pages, setPages] = useState<PageEntry[]>([]);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [curIdx, setCurIdx] = useState(0);
  const studentPdfRef = useRef<HTMLInputElement>(null);

  const [closedPopups, setClosedPopups] = useState<Set<string>>(new Set());
  const [showOcr, setShowOcr] = useState(false);
  const [editableScores, setEditableScores] = useState<Record<string, number>>({});
  const [activeResult, setActiveResult] = useState<AnswerResult | null>(null);
  // manualOverrides: maps q_label -> manually chosen QAItem number (or null = auto)
  const [manualOverrides, setManualOverrides] = useState<Record<string, number | null>>({});

  const handleLoadQA = async () => {
    const qf = qPdfRef.current?.files?.[0];
    const af = aPdfRef.current?.files?.[0];
    if (!qf || !af) { alert('Please select both PDFs.'); return; }
    setLoadingQA(true); setQaError(null);
    const fd = new FormData();
    fd.append('questions_pdf', qf);
    fd.append('answers_pdf', af);
    fd.append('max_marks_per_question', '10');
    try {
      const res = await fetch(`${API}/parse-qa-documents/`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setQaDataset(data.qa_dataset);
    } catch (e: any) { setQaError(e.message); }
    finally { setLoadingQA(false); }
  };

  const handleStudentPdf = useCallback(async (file: File) => {
    setPdfFile(file); setPages([]); setCurIdx(0); setClosedPopups(new Set());

    // Check if the file is an image (not a PDF) — handle it directly as a single page
    const isImage = /\.(jpe?g|png|bmp|gif|webp|tiff?)$/i.test(file.name) ||
                    file.type.startsWith('image/');
    if (isImage) {
      setPages([{ pageIndex: 0, blob: file, url: URL.createObjectURL(file), results: [], loading: false, error: null }]);
      return;
    }

    // For PDFs, use pdfjsLib to render each page as an image
    const pdfjsLib = (window as any).pdfjsLib;
    if (!pdfjsLib) {
      setPages([{ pageIndex: 0, blob: file, url: URL.createObjectURL(file), results: [], loading: false, error: null }]);
      return;
    }
    try {
      const buffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
      const entries: PageEntry[] = [];
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1.5 });
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width; canvas.height = viewport.height;
        await page.render({ canvasContext: canvas.getContext('2d')!, viewport }).promise;
        const blob = await new Promise<Blob>(res => canvas.toBlob(b => res(b!), 'image/jpeg', 0.92));
        entries.push({ pageIndex: i - 1, blob, url: URL.createObjectURL(blob), results: [], loading: false, error: null });
      }
      setPages(entries);
    } catch (err) {
      console.error('PDF parsing failed:', err);
      // Fallback: treat the file as a single page
      setPages([{ pageIndex: 0, blob: file, url: URL.createObjectURL(file), results: [], loading: false, error: null }]);
    }
  }, []);

  const gradeCurrentPage = async () => {
    if (pages.length === 0) return;
    const idx = curIdx;
    setPages(prev => prev.map((p, i) => i === idx ? { ...p, loading: true, error: null } : p));
    setClosedPopups(new Set());
    setActiveResult(null);
    // Note: we do NOT clear manualOverrides — user's Q selection persists across re-analyze

    const page = pages[idx];
    const fd = new FormData();
    fd.append('file', page.blob, `page_${idx + 1}.jpg`);
    fd.append('ref_long', refLong);
    fd.append('ref_short', refShort);
    fd.append('keywords', keywords);
    fd.append('max_marks', maxMarks.toString());
    if (qaDataset.length > 0) fd.append('qa_dataset_json', JSON.stringify(qaDataset));
    // Send manual overrides so backend respects user's question selection
    if (Object.keys(manualOverrides).length > 0) {
      fd.append('manual_q_overrides_json', JSON.stringify(manualOverrides));
    }

    try {
      const res = await fetch(`${API}/grade-page/`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const results: AnswerResult[] = data.results ?? [];

      setPages(prev => prev.map((p, i) => i === idx ? { ...p, results, loading: false } : p));

      // Commit AI scores to the editable grid
      const newScores: Record<string, number> = {};
      for (const r of results) {
        const lm = r.llm_output?.recommended_marks ?? 0;
        const dm = r.deterministic_recommended_marks ?? 0;
        const conf = r.composite_confidence ?? 0;
        const label = r.auto_matched_question != null ? `Q${r.auto_matched_question}` : `Q${r.q_label}`;
        newScores[label] = Number(((conf > 0.6 && lm > 0) ? lm : dm).toFixed(1));
      }
      setEditableScores(prev => ({ ...prev, ...newScores }));
    } catch (e: any) {
      setPages(prev => prev.map((p, i) => i === idx ? { ...p, error: e.message, loading: false } : p));
    }
  };

  const currPage = pages[curIdx] ?? null;
  const popupKey = (pageIdx: number, qLabel: string) => `${pageIdx}-${qLabel}`;

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ─── Header ──────────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div>
          <h1>Ankya AI ⚡</h1>
          <div className="header-sub">Teacher Re-evaluation Dashboard</div>
        </div>
        {pdfFile && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto', fontSize: '0.85rem', color: 'var(--success)' }}>
            <CheckCircle2 size={16} /> {pdfFile.name} · {pages.length} pages
          </div>
        )}
      </header>

      {/* ─── Left Pane: PDF Viewer ──────────────────────────────────────────── */}
      <div className="left-pane">
        {/* Toolbar */}
        {pages.length > 0 && (
          <div className="pdf-toolbar">
            <button className="btn-ghost btn-sm" disabled={curIdx === 0}
              onClick={() => { setCurIdx(i => i - 1); setClosedPopups(new Set()); }}>
              <ChevronLeft size={16} /> Prev
            </button>
            <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>Page {curIdx + 1} of {pages.length}</span>
            <button className="btn-ghost btn-sm" disabled={curIdx === pages.length - 1}
              onClick={() => { setCurIdx(i => i + 1); setClosedPopups(new Set()); }}>
              Next <ChevronRight size={16} />
            </button>
          </div>
        )}

        {/* View Area */}
        <div className="pdf-view-area">
          {!pdfFile ? (
            <div className="drop-zone"
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleStudentPdf(f); }}
              onClick={() => studentPdfRef.current?.click()}>
              <UploadCloud size={52} color="var(--primary)" style={{ marginBottom: 12 }} />
              <h3 style={{ marginBottom: '0.5rem' }}>Drop Student PDF or Image here</h3>
              <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Supports PDF, JPG, PNG — click to browse</p>
              <input type="file" ref={studentPdfRef} accept=".pdf,.jpg,.png" style={{ display: 'none' }}
                onChange={e => { const f = e.target.files?.[0]; if (f) handleStudentPdf(f); }} />
            </div>
          ) : currPage ? (
            <>
              <img src={currPage.url} className="pdf-img" alt={`Page ${curIdx + 1}`} />

              {/* Per-answer floating popups, positioned by y_pct */}
              {currPage.results.map((r, ri) => {
                const key = popupKey(curIdx, r.q_label);
                if (closedPopups.has(key)) return null;
                const topPct = Math.max(2, Math.min(r.y_pct * 100, 55));
                const stackOffset = ri * 5;
                return (
                  <AnswerPopup
                    key={key}
                    res={r}
                    top={`${Math.min(topPct + stackOffset, 70)}%`}
                    onClose={() => setClosedPopups(prev => new Set([...prev, key]))}
                    onSelect={() => setActiveResult(r)}
                    qaDataset={qaDataset}
                    manualQNum={manualOverrides[r.q_label] ?? null}
                    onManualSelect={n => setManualOverrides(prev => ({ ...prev, [r.q_label]: n }))}
                  />
                );
              })}
            </>
          ) : null}
        </div>

        {/* OCR Text Accordion */}
        {currPage && currPage.results.length > 0 && (
          <div style={{ padding: '0 1rem', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', padding: '0.4rem 0', border: 'none', background: 'transparent', color: 'var(--muted)', fontSize: '0.8rem' }}
              onClick={() => setShowOcr(v => !v)}>
              <span>📄 Extracted OCR Text</span>
              {showOcr ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </div>
            {showOcr && <div className="ocr-box">{currPage.results[0].full_page_text || 'No text extracted.'}</div>}
          </div>
        )}

        {/* Error */}
        {currPage?.error && (
          <div className="error-box" style={{ margin: '0 1rem 0.5rem' }}>
            <AlertCircle size={14} style={{ marginRight: 4 }} />{currPage.error}
          </div>
        )}

        {/* Analyze Button */}
        {pages.length > 0 && (
          <button className="analyze-btn" onClick={gradeCurrentPage} disabled={currPage?.loading}>
            {currPage?.loading ? <><Loader2 size={18} className="spinner" />AI is evaluating...</> : <><Sparkles size={18} />Analyze Page {curIdx + 1}</>}
          </button>
        )}

        {/* Thumbnail Strip */}
        {pages.length > 1 && (
          <div className="thumb-strip">
            {pages.map((p, i) => (
              <div key={i} className={`thumb ${i === curIdx ? 'active' : ''}`}
                onClick={() => { setCurIdx(i); setClosedPopups(new Set()); }}>
                <img src={p.url} alt={`pg ${i + 1}`} />
                <span className="thumb-label">pg {i + 1}</span>
                {p.results.length > 0 && <span className="thumb-graded" />}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ─── Right Pane ──────────────────────────────────────────────────────── */}
      <div className="right-pane">

        {/* Solutions */}
        <div className="right-section">
          <div className="panel-title"><BookOpen size={16} />Solutions for trained model</div>
          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '3px' }}>Questions PDF</div>
              <input type="file" ref={qPdfRef} accept=".pdf" style={{ display: 'none' }} />
              <button className="btn-ghost btn-sm" style={{ width: '100%' }} onClick={() => qPdfRef.current?.click()}>
                {qPdfRef.current?.files?.[0]?.name ?? 'Choose…'}
              </button>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '3px' }}>Answer Key PDF</div>
              <input type="file" ref={aPdfRef} accept=".pdf" style={{ display: 'none' }} />
              <button className="btn-ghost btn-sm" style={{ width: '100%' }} onClick={() => aPdfRef.current?.click()}>
                {aPdfRef.current?.files?.[0]?.name ?? 'Choose…'}
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button className="btn-sm" onClick={handleLoadQA} disabled={loadingQA}>
                {loadingQA ? <Loader2 size={14} className="spinner" /> : 'Extract'}
              </button>
            </div>
          </div>
          {qaDataset.length > 0 && <div style={{ fontSize: '0.8rem', color: 'var(--success)' }}><CheckCircle2 size={13} style={{ marginRight: 4 }} />{qaDataset.length} questions parsed — auto-grading enabled</div>}
          {qaError && <div className="error-box">{qaError}</div>}
        </div>

        {/* Rubric Section */}
        <div className="right-section">
          <div className="panel-title">📋 Rubric</div>

          {/* If a graded answer is selected, show its matched rubric */}
          {activeResult ? (
            <RubricPanel
              activeResult={activeResult}
              qaDataset={qaDataset}
              manualOverrides={manualOverrides}
              setManualOverrides={setManualOverrides}
              onBack={() => setActiveResult(null)}
            />
          ) : (
            <div>
              <p style={{ fontSize: '0.78rem', color: 'var(--muted)', marginBottom: '0.75rem' }}>
                {qaDataset.length > 0
                  ? 'Click a popup to preview its rubric, or override manually below.'
                  : 'Enter rubric manually since no Q&A dataset is loaded.'}
              </p>
              <div className="form-group">
                <label>Reference Answer</label>
                <textarea rows={2} value={refLong} onChange={e => setRefLong(e.target.value)} />
              </div>
              <div style={{ display: 'flex', gap: '0.6rem' }}>
                <div className="form-group" style={{ flex: 2 }}>
                  <label>Keywords</label>
                  <input value={keywords} onChange={e => setKeywords(e.target.value)} />
                </div>
                <div className="form-group" style={{ flex: 1 }}>
                  <label>Max Marks</label>
                  <input type="number" value={maxMarks} onChange={e => setMaxMarks(Number(e.target.value))} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Score Grid */}
        <div className="right-section" style={{ flex: 1 }}>
          <div className="panel-title">📊 Answer Score from Model</div>
          <p style={{ fontSize: '0.75rem', color: 'var(--warning)', marginBottom: '0.5rem' }}>* Editable for the teacher</p>
          {Object.keys(editableScores).length === 0 ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--muted)', fontStyle: 'italic', marginTop: '1rem', textAlign: 'center' }}>Analyze a page to generate scores…</p>
          ) : (
            <table className="score-table">
              <thead>
                <tr>
                  {Object.keys(editableScores).map(k => <th key={k}>{k}</th>)}
                  <th style={{ color: 'var(--primary)' }}>Total</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  {Object.entries(editableScores).map(([k, v]) => (
                    <td key={k}>
                      <input type="number" className="score-cell" value={v} step={0.5}
                        onChange={e => setEditableScores(prev => ({ ...prev, [k]: parseFloat(e.target.value) || 0 }))} />
                    </td>
                  ))}
                  <td style={{ fontWeight: 800, fontSize: '1.3rem', color: 'var(--success)' }}>
                    {Object.values(editableScores).reduce((a, b) => a + b, 0).toFixed(1)}
                  </td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
