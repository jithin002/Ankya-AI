import React, { useState, useRef } from 'react';
import { UploadCloud, Sparkles, ChevronDown, ChevronUp, AlertCircle, CheckCircle2, FileText, Wand2 } from 'lucide-react';
import './App.css';

interface GradingResult {
  page_num?: number;
  component_scores?: {
    keyword_pct?: number;
    grammar_pct?: number;
    semantic_pct?: number;
    presentation_pct?: number;
  };
  llm_output?: {
    recommended_marks?: number;
    explanation?: string;
  };
  deterministic_final_pct?: number;
  deterministic_recommended_marks?: number;
  composite_confidence?: number;
  student_text?: string;
  [key: string]: any;
}

function App() {
  const [question, setQuestion] = useState("Explain Newton's first law of motion.");
  const [refLong, setRefLong] = useState("Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force (inertia).");
  const [refShort, setRefShort] = useState("inertia, object at rest stays at rest, object in motion stays in motion, external force");
  const [keywords, setKeywords] = useState("Newton, Inertia, Force, Motion, Rest");
  const [maxMarks, setMaxMarks] = useState<number>(10);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<GradingResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Auto Rubric State
  const [showAutoRubric, setShowAutoRubric] = useState(false);
  const [akFile, setAkFile] = useState<File | null>(null);
  const [qpFile, setQpFile] = useState<File | null>(null);
  const [generatingRubric, setGeneratingRubric] = useState(false);
  const akFileInputRef = useRef<HTMLInputElement>(null);
  const qpFileInputRef = useRef<HTMLInputElement>(null);

  const [showOcrPages, setShowOcrPages] = useState<{ [key: number]: boolean }>({});
  const [showJsonPages, setShowJsonPages] = useState<{ [key: number]: boolean }>({});

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      if (selectedFile.type.includes('image')) {
        setPreview(URL.createObjectURL(selectedFile));
      } else {
        setPreview(null); // PDF preview not supported directly without library
      }
      setResults(null);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      if (selectedFile.type.includes('image')) {
        setPreview(URL.createObjectURL(selectedFile));
      } else {
        setPreview(null);
      }
      setResults(null);
      setError(null);
    }
  };

  const handleGrade = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);
    formData.append('ref_long', refLong);
    formData.append('ref_short', refShort);
    formData.append('keywords', keywords);
    formData.append('max_marks', maxMarks.toString());

    try {
      const response = await fetch('http://localhost:8000/grade-image/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred during grading.');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateRubric = async () => {
    if (!akFile) {
      alert("Please upload an Answer Key to generate a rubric.");
      return;
    }

    setGeneratingRubric(true);
    setError(null);

    const formData = new FormData();
    formData.append('question', question);
    formData.append('answer_key', akFile);
    if (qpFile) {
      formData.append('question_paper', qpFile);
    }

    try {
      const response = await fetch('http://localhost:8000/generate-rubric/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setRefLong(data.reference_long || "");
      setRefShort(data.reference_short ? data.reference_short.join(", ") : "");

      const kws = data.keywords || [];
      const kwString = kws.map((k: any) => k.term).join(", ");
      setKeywords(kwString);

      if (data.max_marks) setMaxMarks(data.max_marks);

      alert("Rubric generated successfully!");
      setShowAutoRubric(false);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred during rubric generation.');
    } finally {
      setGeneratingRubric(false);
    }
  };

  const renderMetrics = () => {
    if (!results || results.length === 0) return null;

    let totalMarks = 0;
    const totalMaxMarks = maxMarks * results.length;

    results.forEach(result => {
      const llmRes = result.llm_output || {};
      const recMarks = result.deterministic_recommended_marks || 0;
      const llmMarks = llmRes.recommended_marks || 0;
      const conf = result.composite_confidence || 0;
      const displayMarks = (conf > 0.6 && llmMarks > 0) ? llmMarks : recMarks;
      totalMarks += displayMarks;
    });

    const avgPct = totalMaxMarks > 0 ? (totalMarks / totalMaxMarks) * 100 : 0;

    return (
      <div className="glass-panel" style={{ marginTop: '2rem' }}>
        <h2 className="panel-title"><Sparkles size={24} color="var(--primary-color)" /> Grading Results</h2>

        <div className="score-card">
          <div className="score-label">Total Grade</div>
          <div className="score-value">{totalMarks.toFixed(1)} / {totalMaxMarks}</div>
          <div className="score-percent">({avgPct.toFixed(1)}%)</div>
        </div>

        {results.map((result, idx) => {
          const pageNum = result.page_num || (idx + 1);
          const scores = result.component_scores || {};
          const llmRes = result.llm_output || {};
          return (
            <div key={idx} style={{ marginTop: '2rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>Page {pageNum} Breakdown</h3>

              <div className="results-grid">
                <div className="metric">
                  <div className="metric-header">
                    <span>Keywords</span>
                    <span>{Math.round(scores.keyword_pct || 0)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${scores.keyword_pct || 0}%` }}></div>
                  </div>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Grammar</span>
                    <span>{Math.round(scores.grammar_pct || 0)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${scores.grammar_pct || 0}%` }}></div>
                  </div>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Semantic</span>
                    <span>{Math.round(scores.semantic_pct || 0)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${scores.semantic_pct || 0}%` }}></div>
                  </div>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Presentation</span>
                    <span>{Math.round(scores.presentation_pct || 0)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${scores.presentation_pct || 0}%` }}></div>
                  </div>
                </div>

                <div className="feedback-box">
                  <div className="feedback-title">
                    🤖 AI Feedback
                  </div>
                  <div>{llmRes.explanation || "No detailed feedback generated."}</div>
                </div>
              </div>

              <div className="expandable-section">
                <div className="expandable-header" onClick={() => setShowOcrPages({ ...showOcrPages, [idx]: !showOcrPages[idx] })}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FileText size={18} /> Extracted Text (OCR)
                  </span>
                  {showOcrPages[idx] ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
                {showOcrPages[idx] && (
                  <div className="expandable-content" style={{ whiteSpace: 'pre-wrap' }}>
                    {result.student_text || "No text extracted."}
                  </div>
                )}
              </div>

              <div className="expandable-section" style={{ marginTop: '1rem' }}>
                <div className="expandable-header" onClick={() => setShowJsonPages({ ...showJsonPages, [idx]: !showJsonPages[idx] })}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {'{}'} Raw Debug Data (JSON)
                  </span>
                  {showJsonPages[idx] ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
                {showJsonPages[idx] && (
                  <div className="expandable-content" style={{ whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(result, null, 2)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="app-container">
      <header>
        <h1>Intelligent Classroom Grader</h1>
        <p>Upload a student's handwritten answer to grade it automatically using <strong>OCR + AI</strong>.</p>
      </header>

      <aside>
        <div className="glass-panel" style={{ marginBottom: '1rem', backgroundColor: 'var(--panel-bg)' }}>
          <div
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
            onClick={() => setShowAutoRubric(!showAutoRubric)}
          >
            <h2 className="panel-title" style={{ margin: 0 }}><Wand2 size={20} style={{ marginRight: '8px' }} /> Auto-Generate Rubric</h2>
            {showAutoRubric ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>

          {showAutoRubric && (
            <div style={{ marginTop: '1rem' }}>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Upload an Answer Key and/or Question Paper to formulate the rubric.</p>

              <div style={{ marginTop: '1rem' }}>
                <label style={{ fontSize: '0.9rem', fontWeight: 600 }}>1. Target Question</label>
                <textarea
                  className="form-group"
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  style={{ width: '100%', marginTop: '0.5rem', background: 'rgba(255, 255, 255, 0.05)', color: 'white' }}
                  placeholder="Type the question here..."
                />
              </div>

              <div style={{ marginTop: '1rem' }}>
                <label style={{ fontSize: '0.9rem', fontWeight: 600 }}>2. Answer Key (PDF/TXT)</label>
                <input type="file" ref={akFileInputRef} onChange={(e) => setAkFile(e.target.files?.[0] || null)} style={{ display: 'none' }} accept=".pdf,.txt" />
                <div className="btn-secondary" style={{ marginTop: '0.5rem', textAlign: 'center', cursor: 'pointer' }} onClick={() => akFileInputRef.current?.click()}>
                  {akFile ? akFile.name : "Choose File"}
                </div>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <label style={{ fontSize: '0.9rem', fontWeight: 600 }}>3. Question Paper [Optional]</label>
                <input type="file" ref={qpFileInputRef} onChange={(e) => setQpFile(e.target.files?.[0] || null)} style={{ display: 'none' }} accept=".pdf,.txt" />
                <div className="btn-secondary" style={{ marginTop: '0.5rem', textAlign: 'center', cursor: 'pointer' }} onClick={() => qpFileInputRef.current?.click()}>
                  {qpFile ? qpFile.name : "Choose File"}
                </div>
              </div>

              <button
                className="btn-primary"
                style={{ width: '100%', marginTop: '1.5rem' }}
                onClick={handleGenerateRubric}
                disabled={generatingRubric}
              >
                {generatingRubric ? "Generating..." : "Generate Rubric"}
              </button>
            </div>
          )}
        </div>

        <div className="glass-panel">
          <h2 className="panel-title">⚙️ Grading Rubric</h2>

          <div className="form-group">
            <label>Question</label>
            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Reference Answer (Long)</label>
            <textarea
              rows={4}
              value={refLong}
              onChange={(e) => setRefLong(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Key Points (comma separated)</label>
            <textarea
              rows={2}
              value={refShort}
              onChange={(e) => setRefShort(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Keywords (comma separated)</label>
            <input
              type="text"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>Max Marks</label>
            <input
              type="number"
              min="1"
              max="100"
              value={maxMarks}
              onChange={(e) => setMaxMarks(Number(e.target.value))}
            />
          </div>
        </div>
      </aside>

      <main>
        <div className="glass-panel">
          <h2 className="panel-title">1. Upload Answer</h2>

          <div
            className="upload-area"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud className="upload-icon" />
            <h3>Choose an image/PDF or drag & drop</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>JPG, JPEG, PNG, PDF</p>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept=".jpg,.jpeg,.png,.pdf"
              onChange={handleFileChange}
            />
          </div>

          {file && !preview && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', margin: '1rem 0', color: 'var(--success)' }}>
              <CheckCircle2 size={20} /> {file.name} ready for grading
            </div>
          )}

          {preview && (
            <div className="preview-container">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '1rem', color: 'var(--success)' }}>
                <CheckCircle2 size={20} /> Image selected
              </div>
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          <button
            className="btn-primary"
            disabled={!file || loading}
            onClick={handleGrade}
          >
            {loading ? (
              <><UploadCloud className="spinner" size={20} /> Processing...</>
            ) : (
              <><Sparkles size={20} /> Grade Answer</>
            )}
          </button>

          {error && (
            <div className="error-box">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold', marginBottom: '4px' }}>
                <AlertCircle size={18} /> Error
              </div>
              {error}
            </div>
          )}
        </div>

        {renderMetrics()}
      </main>
    </div>
  );
}

export default App;
