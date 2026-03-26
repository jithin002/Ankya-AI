import React, { useState, useRef, useCallback } from 'react';
import { UploadCloud, Sparkles, ChevronDown, ChevronUp, AlertCircle, CheckCircle2, FileText, BookOpen, Loader2 } from 'lucide-react';
import './App.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Rubric {
  max_marks: number;
  reference_long: string;
  reference_short: string[];
  keywords: { term: string; weight: number }[];
}

interface QAItem {
  number: number;
  section: string;
  question: string;
  answer: string;
  rubric: Rubric;
}

interface GradingResult {
  page_num?: number;
  component_scores?: {
    keyword_pct?: number;
    grammar_pct?: number;
    semantic_pct?: number;
    presentation_pct?: number;
  };
  llm_output?: { recommended_marks?: number; explanation?: string };
  deterministic_recommended_marks?: number;
  composite_confidence?: number;
  student_text?: string;
  [key: string]: any;
}

interface PageEntry {
  pageIndex: number;
  blob: Blob;
  url: string;
  result: GradingResult | null;
  loading: boolean;
  error: string | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const API = 'http://localhost:8000';

function ScoreBar({ label, pct }: { label: string; pct: number }) {
  return (
    <div className="metric">
      <div className="metric-header">
        <span>{label}</span>
        <span>{Math.round(pct)}%</span>
      </div>
      <div className="progress-bar-bg">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  // ── QA dataset ──
  const [qaDataset, setQaDataset] = useState<QAItem[]>([]);
  const [selectedQ, setSelectedQ] = useState<QAItem | null>(null);
  const [loadingQA, setLoadingQA] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);
  const qPdfRef = useRef<HTMLInputElement>(null);
  const aPdfRef = useRef<HTMLInputElement>(null);

  // ── Manual rubric ──
  const [question, setQuestion] = useState("Explain Newton's first law of motion.");
  const [refLong, setRefLong] = useState("Newton's first law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an unbalanced force.");
  const [refShort, setRefShort] = useState("inertia, object at rest stays at rest, external force");
  const [keywords, setKeywords] = useState("Newton, Inertia, Force, Motion, Rest");
  const [maxMarks, setMaxMarks] = useState(10);

  // ── Student PDF pages ──
  const [pages, setPages] = useState<PageEntry[]>([]);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const studentPdfRef = useRef<HTMLInputElement>(null);

  // ── Collapsible panels ──
  const [showResult, setShowResult] = useState<{ [k: number]: boolean }>({});
  const [showOcr, setShowOcr] = useState<{ [k: number]: boolean }>({});

  // ── Load QA PDFs ──────────────────────────────────────────────────────────
  const handleLoadQA = async () => {
    const qFile = qPdfRef.current?.files?.[0];
    const aFile = aPdfRef.current?.files?.[0];
    if (!qFile || !aFile) { alert('Please select both a Questions PDF and an Answers PDF.'); return; }

    setLoadingQA(true); setQaError(null);
    const fd = new FormData();
    fd.append('questions_pdf', qFile);
    fd.append('answers_pdf', aFile);
    fd.append('max_marks_per_question', '10');

    try {
      const res = await fetch(`${API}/parse-qa-documents/`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setQaDataset(data.qa_dataset);
      setSelectedQ(null);
    } catch (e: any) {
      setQaError(e.message);
    } finally {
      setLoadingQA(false);
    }
  };

  const applyRubric = (item: QAItem) => {
    setSelectedQ(item);
    setQuestion(item.question);
    setRefLong(item.rubric.reference_long);
    setRefShort(item.rubric.reference_short.join(', '));
    setKeywords(item.rubric.keywords.map(k => k.term).join(', '));
    setMaxMarks(item.rubric.max_marks);
    setPages([]); // reset pages when selecting a new question
  };

  // ── Load Student PDF as page images ──────────────────────────────────────
  const handleStudentPdf = useCallback(async (file: File) => {
    setPdfFile(file);
    setPages([]);

    // Use pdfjs-dist (loaded via CDN in the HTML) to render pages
    const pdfjsLib = (window as any).pdfjsLib;
    if (!pdfjsLib) {
      // Fallback: just treat the whole PDF as one file
      const entry: PageEntry = {
        pageIndex: 0, blob: file, url: URL.createObjectURL(file),
        result: null, loading: false, error: null
      };
      setPages([entry]);
      return;
    }

    const buffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
    const entries: PageEntry[] = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.5 });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width; canvas.height = viewport.height;
      await page.render({ canvasContext: canvas.getContext('2d')!, viewport }).promise;

      const blob = await new Promise<Blob>((resolve) =>
        canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.92)
      );
      entries.push({
        pageIndex: i - 1,
        blob,
        url: URL.createObjectURL(blob),
        result: null, loading: false, error: null
      });
    }
    setPages(entries);
  }, []);

  // ── Grade a single page ─────────────────────────────────────────────────
  const gradePage = async (idx: number) => {
    setPages(prev => prev.map((p, i) => i === idx ? { ...p, loading: true, error: null } : p));
    const page = pages[idx];

    const fd = new FormData();
    fd.append('file', page.blob, `page_${idx + 1}.jpg`);
    fd.append('question', question);
    fd.append('ref_long', refLong);
    fd.append('ref_short', refShort);
    fd.append('keywords', keywords);
    fd.append('max_marks', maxMarks.toString());

    // Send full qa_dataset so backend can auto-select the correct rubric via RAG
    if (qaDataset.length > 0) {
      fd.append('qa_dataset_json', JSON.stringify(qaDataset));
    }

    try {
      const res = await fetch(`${API}/grade-page/`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const result = data.results?.[0] ?? null;
      setPages(prev => prev.map((p, i) => i === idx ? { ...p, result, loading: false } : p));
      setShowResult(prev => ({ ...prev, [idx]: true }));
    } catch (e: any) {
      setPages(prev => prev.map((p, i) => i === idx ? { ...p, error: e.message, loading: false } : p));
    }
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      <header>
        <h1>Intelligent Classroom Grader</h1>
        <p>Upload Question &amp; Answer PDFs to set a rubric, then grade student answer sheets <strong>page by page</strong>.</p>
      </header>

      {/* ─── Sidebar ─────────────────────────────────────────────────────── */}
      <aside>
        {/* --- Load QA Documents --- */}
        <div className="glass-panel" style={{ marginBottom: '1rem' }}>
          <h2 className="panel-title"><BookOpen size={20} style={{ marginRight: 8 }} />Load Q&amp;A Documents</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
            Upload a numbered Questions PDF and its matching Answer Key PDF.
          </p>

          <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Questions PDF</label>
          <input type="file" ref={qPdfRef} accept=".pdf" style={{ display: 'none' }} />
          <div className="btn-secondary" style={{ textAlign: 'center', cursor: 'pointer', marginTop: '0.4rem', marginBottom: '0.75rem' }}
            onClick={() => qPdfRef.current?.click()}>
            {qPdfRef.current?.files?.[0]?.name ?? 'Choose Questions PDF'}
          </div>

          <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Answer Key PDF</label>
          <input type="file" ref={aPdfRef} accept=".pdf" style={{ display: 'none' }} />
          <div className="btn-secondary" style={{ textAlign: 'center', cursor: 'pointer', marginTop: '0.4rem' }}
            onClick={() => aPdfRef.current?.click()}>
            {aPdfRef.current?.files?.[0]?.name ?? 'Choose Answer Key PDF'}
          </div>

          <button className="btn-primary" style={{ width: '100%', marginTop: '1rem' }}
            onClick={handleLoadQA} disabled={loadingQA}>
            {loadingQA ? <><Loader2 size={16} className="spinner" style={{ marginRight: 6 }} />Extracting...</> : 'Extract Q&A'}
          </button>

          {qaError && <div className="error-box" style={{ marginTop: '0.75rem' }}>{qaError}</div>}

          {/* Question list */}
          {qaDataset.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                {qaDataset.length} questions found. Click to set rubric:
              </div>
              <div style={{ maxHeight: '220px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {qaDataset.map(item => (
                  <div key={item.number}
                    onClick={() => applyRubric(item)}
                    style={{
                      padding: '0.5rem 0.75rem', borderRadius: '8px', cursor: 'pointer', fontSize: '0.85rem',
                      background: selectedQ?.number === item.number ? 'var(--primary-color)' : 'rgba(255,255,255,0.05)',
                      color: selectedQ?.number === item.number ? '#fff' : 'var(--text-muted)',
                      transition: 'background 0.2s'
                    }}>
                    <strong>Q{item.number}</strong> ({item.section}) — {item.question.slice(0, 50)}{item.question.length > 50 ? '…' : ''}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* --- Manual Rubric --- */}
        <div className="glass-panel">
          <h2 className="panel-title">⚙️ Grading Rubric</h2>
          {selectedQ && (
            <div style={{ fontSize: '0.8rem', color: 'var(--primary-color)', marginBottom: '0.75rem' }}>
              <CheckCircle2 size={14} style={{ marginRight: 4 }} />Auto-filled from Q{selectedQ.number}
            </div>
          )}

          <div className="form-group">
            <label>Question</label>
            <textarea rows={3} value={question} onChange={e => setQuestion(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Reference Answer (Long)</label>
            <textarea rows={4} value={refLong} onChange={e => setRefLong(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Key Points (comma separated)</label>
            <textarea rows={2} value={refShort} onChange={e => setRefShort(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Keywords (comma separated)</label>
            <input type="text" value={keywords} onChange={e => setKeywords(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Max Marks</label>
            <input type="number" min={1} max={100} value={maxMarks} onChange={e => setMaxMarks(Number(e.target.value))} />
          </div>
        </div>
      </aside>

      {/* ─── Main ────────────────────────────────────────────────────────── */}
      <main>
        {/* --- Upload Student PDF --- */}
        <div className="glass-panel">
          <h2 className="panel-title">1. Upload Student Answer Sheet</h2>
          <div className="upload-area"
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleStudentPdf(f); }}
            onClick={() => studentPdfRef.current?.click()}>
            <UploadCloud className="upload-icon" />
            <h3>Choose a PDF or drag &amp; drop</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>PDF — pages will appear below for individual grading</p>
            <input type="file" ref={studentPdfRef} accept=".pdf,.jpg,.jpeg,.png" style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) handleStudentPdf(f); }} />
          </div>

          {pdfFile && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: '1rem', color: 'var(--success)' }}>
              <CheckCircle2 size={18} /> {pdfFile.name} — {pages.length} page(s) loaded
            </div>
          )}
        </div>

        {/* --- Page-by-Page Grading --- */}
        {pages.length > 0 && (
          <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
            <h2 className="panel-title">2. Grade Page by Page</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Review each page, then click <strong>Analyze</strong> to grade it against the rubric on the left.
            </p>

            {pages.map((page, idx) => {
              const res = page.result;
              const scores = res?.component_scores ?? {};
              const llm = res?.llm_output ?? {};
              const recMarks = res?.deterministic_recommended_marks ?? 0;
              const conf = res?.composite_confidence ?? 0;
              const llmMarks = llm.recommended_marks ?? 0;
              const displayMarks = (conf > 0.6 && llmMarks > 0) ? llmMarks : recMarks;

              return (
                <div key={idx} style={{ marginBottom: '2rem', borderBottom: '1px solid var(--border)', paddingBottom: '2rem' }}>
                  {/* Page image + controls */}
                  <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                    <div style={{ flex: '0 0 220px' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                        Page {idx + 1}
                      </div>
                      <img src={page.url} alt={`Page ${idx + 1}`}
                        style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border)', objectFit: 'cover' }} />
                      <button className="btn-primary" style={{ width: '100%', marginTop: '0.75rem' }}
                        onClick={() => gradePage(idx)} disabled={page.loading}>
                        {page.loading
                          ? <><Loader2 size={16} className="spinner" style={{ marginRight: 6 }} />Analyzing...</>
                          : <><Sparkles size={16} style={{ marginRight: 6 }} />Analyze Page {idx + 1}</>}
                      </button>
                      {page.error && (
                        <div className="error-box" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
                          <AlertCircle size={14} style={{ marginRight: 4 }} />{page.error}
                        </div>
                      )}
                    </div>

                    {/* Results panel */}
                    {res && (
                      <div style={{ flex: 1, minWidth: '260px' }}>
                    {/* Auto-matched Q badge */}
                    {res && (
                      <div style={{ marginBottom: '0.75rem' }}>
                        {res.auto_matched_question != null ? (
                          <span style={{
                            background: 'rgba(0,220,130,0.15)', color: 'var(--success)',
                            border: '1px solid var(--success)', borderRadius: '20px',
                            padding: '3px 12px', fontSize: '0.78rem', fontWeight: 600
                          }}>
                            🎯 RAG Auto-matched: Q{res.auto_matched_question}
                          </span>
                        ) : (
                          <span style={{
                            background: 'rgba(255,200,0,0.1)', color: '#f0c040',
                            border: '1px solid #f0c040', borderRadius: '20px',
                            padding: '3px 12px', fontSize: '0.78rem', fontWeight: 600
                          }}>
                            ⚙️ Manual Rubric Used
                          </span>
                        )}
                      </div>
                    )}
                        <div className="score-card" style={{ marginBottom: '1rem', padding: '1rem' }}>
                          <div className="score-label">Page {idx + 1} Score</div>
                          <div className="score-value">{displayMarks.toFixed(1)} / {maxMarks}</div>
                          <div className="score-percent">({maxMarks > 0 ? ((displayMarks / maxMarks) * 100).toFixed(1) : 0}%)</div>
                        </div>

                        <div className="results-grid">
                          <ScoreBar label="Keywords" pct={scores.keyword_pct ?? 0} />
                          <ScoreBar label="Grammar" pct={scores.grammar_pct ?? 0} />
                          <ScoreBar label="Semantic" pct={scores.semantic_pct ?? 0} />
                          <ScoreBar label="Presentation" pct={scores.presentation_pct ?? 0} />
                        </div>

                        {llm.explanation && (
                          <div className="feedback-box" style={{ marginTop: '1rem' }}>
                            <div className="feedback-title">🤖 AI Feedback</div>
                            <div>{llm.explanation}</div>
                          </div>
                        )}

                        {/* OCR text expandable */}
                        <div className="expandable-section" style={{ marginTop: '1rem' }}>
                          <div className="expandable-header" onClick={() => setShowOcr(p => ({ ...p, [idx]: !p[idx] }))}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <FileText size={16} /> Extracted Text (OCR)
                            </span>
                            {showOcr[idx] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                          </div>
                          {showOcr[idx] && (
                            <div className="expandable-content" style={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>
                              {res.student_text || 'No text extracted.'}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Overall summary if multiple pages graded */}
            {pages.filter(p => p.result).length > 1 && (() => {
              const graded = pages.filter(p => p.result);
              const total = graded.reduce((s, p) => {
                const res = p.result!;
                const conf = res.composite_confidence ?? 0;
                const llmM = res.llm_output?.recommended_marks ?? 0;
                const detM = res.deterministic_recommended_marks ?? 0;
                return s + ((conf > 0.6 && llmM > 0) ? llmM : detM);
              }, 0);
              const max = maxMarks * graded.length;
              return (
                <div className="score-card" style={{ marginTop: '1rem' }}>
                  <div className="score-label">Overall Score ({graded.length} pages graded)</div>
                  <div className="score-value">{total.toFixed(1)} / {max}</div>
                  <div className="score-percent">({max > 0 ? ((total / max) * 100).toFixed(1) : 0}%)</div>
                </div>
              );
            })()}
          </div>
        )}
      </main>
    </div>
  );
}
