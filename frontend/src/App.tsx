import React, { useState, useRef } from 'react';
import { UploadCloud, FileImage, Sparkles, ChevronDown, ChevronUp, AlertCircle, CheckCircle2 } from 'lucide-react';
import './App.css';

interface GradingResult {
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
  const [result, setResult] = useState<GradingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [showOcr, setShowOcr] = useState(false);
  const [showJson, setShowJson] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
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
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleGrade = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);
    formData.append('ref_long', refLong);
    formData.append('ref_short', refShort);
    formData.append('keywords', keywords);
    formData.append('max_marks', maxMarks.toString());

    try {
      // Assuming FastAPI is running on localhost:8000
      const response = await fetch('http://localhost:8000/grade-image/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred during grading.');
    } finally {
      setLoading(false);
    }
  };

  const renderMetrics = () => {
    if (!result) return null;

    const scores = result.component_scores || {};
    const llmRes = result.llm_output || {};
    
    let displayMarks = result.deterministic_recommended_marks || 0;
    const llmMarks = llmRes.recommended_marks || 0;
    const conf = result.composite_confidence || 0;

    if (conf > 0.6 && llmMarks > 0) {
      displayMarks = llmMarks;
    }

    const displayPct = (displayMarks / maxMarks) * 100;

    return (
      <div className="glass-panel" style={{ marginTop: '2rem' }}>
        <h2 className="panel-title"><Sparkles size={24} color="var(--primary-color)" /> Grading Results</h2>
        
        <div className="score-card">
          <div className="score-label">Final Grade</div>
          <div className="score-value">{displayMarks.toFixed(1)} / {maxMarks}</div>
          <div className="score-percent">({displayPct.toFixed(1)}%)</div>
        </div>

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
              <span>Semantic Meaning</span>
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
          <div className="expandable-header" onClick={() => setShowOcr(!showOcr)}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileImage size={18} /> Extracted Text (OCR)
            </span>
            {showOcr ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>
          {showOcr && (
            <div className="expandable-content">
              {result.student_text || "No text extracted."}
            </div>
          )}
        </div>

        <div className="expandable-section" style={{ marginTop: '1rem' }}>
          <div className="expandable-header" onClick={() => setShowJson(!showJson)}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {'{}'} Raw Debug Data (JSON)
            </span>
            {showJson ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>
          {showJson && (
            <div className="expandable-content">
              {JSON.stringify(result, null, 2)}
            </div>
          )}
        </div>
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
            <h3>Choose an image or drag & drop</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>JPG, JPEG, PNG</p>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept=".jpg,.jpeg,.png"
              onChange={handleFileChange}
            />
          </div>

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
