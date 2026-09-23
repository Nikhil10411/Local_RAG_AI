import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { streamMessage, submitFeedback } from "./services/api";
import "./App.css";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello! 👋 I'm your local AI engineering partner. You can chat, upload documents/images (OCR enabled), or transform file formats offline.",
      id: null,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [showConverter, setShowConverter] = useState(false);
  const [convertFile, setConvertFile] = useState(null);
  const [targetFormat, setTargetFormat] = useState("docx");
  const [convertedUrl, setConvertedUrl] = useState(null);
  const [convertedFileName, setConvertedFileName] = useState(null);
  const [isConverting, setIsConverting] = useState(false);

  // New Generation Controls
  const [engineMode, setEngineMode] = useState("hybrid"); // 'chroma_only' | 'hybrid' | 'model_only'
  const [detailLevel, setDetailLevel] = useState("detailed"); // 'short' | 'detailed' | 'full'

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const showToast = (text) => {
    setToast(text);
    setTimeout(() => setToast(null), 3500);
  };

  const fetchStats = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/system/stats");
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch {
      // Backend temporarily unreachable
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleInputChange = (e) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userText },
      {
        role: "assistant",
        content: "",
        id: null,
        accuracy: undefined,
        grounding: undefined,
        tokenSpeed: undefined,
        modeUsed: engineMode,
        depthUsed: detailLevel,
      },
    ]);
    setLoading(true);

    try {
      await streamMessage(
        userText,
        (currentText) => {
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              lastMsg.content = currentText;
            }
            return updated;
          });
        },
        (metadata) => {
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              lastMsg.id = metadata.message_id;
              lastMsg.accuracy = metadata.retrieval_accuracy;
              lastMsg.grounding = metadata.grounding_score;
              lastMsg.tokenSpeed = metadata.tokens_per_second;
            }
            return updated;
          });
          fetchStats();
        },
        engineMode,
        detailLevel
      );
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg && lastMsg.role === "assistant") {
          lastMsg.content =
            "⚠️ Unable to reach the local backend. Please verify that Ollama and FastAPI are running.";
        }
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFeedback = async (messageId, val) => {
    if (!messageId) return;
    try {
      await submitFeedback(messageId, val);
      showToast(
        val === 1 ? "✨ Feedback saved (Helpful)" : "🎯 Calibration rule recorded"
      );
      fetchStats();
    } catch {
      showToast("❌ Could not save feedback.");
    }
  };

  const copyCode = (codeText) => {
    navigator.clipboard.writeText(codeText);
    showToast("📋 Code copied to clipboard!");
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    showToast(`⏳ Ingesting ${file.name} (OCR / Parsing)...`);
    try {
      const res = await fetch("http://localhost:8000/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload error");
      showToast(`✅ Indexed ${data.chunks_indexed} chunks from ${file.name}`);
      fetchStats();
    } catch (err) {
      showToast(`❌ Upload failed: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  };

  const triggerBrowserDownload = (downloadUrl, fileName) => {
    const anchor = document.createElement("a");
    anchor.href = downloadUrl;
    anchor.setAttribute("download", fileName);
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  };

  const handleConvertSubmit = async (e) => {
    e.preventDefault();
    if (!convertFile) {
      showToast("Select a file first");
      return;
    }

    setIsConverting(true);
    const formData = new FormData();
    formData.append("file", convertFile);
    formData.append("target_format", targetFormat);

    showToast(`Transforming ${convertFile.name} to ${targetFormat.toUpperCase()}...`);
    try {
      const res = await fetch("http://localhost:8000/api/convert", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Conversion error");

      setConvertedUrl(data.download_url);
      setConvertedFileName(data.converted_file);

      triggerBrowserDownload(data.download_url, data.converted_file);
      showToast(`🎉 Converted to ${data.converted_file}! Starting download...`);
      fetchStats();
    } catch (err) {
      showToast(`❌ Conversion error: ${err.message}`);
    } finally {
      setIsConverting(false);
    }
  };

  return (
    <div className="app-shell">
      {toast && <div className="toast">{toast}</div>}

      <input
        type="file"
        ref={fileInputRef}
        accept=".pdf,.txt,.docx,.pptx,.xlsx,.csv,.png,.jpg,.jpeg"
        style={{ display: "none" }}
        onChange={handleFileUpload}
      />

      {/* Sidebar: Navigation & Live Metrics */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <h1>Local AI System</h1>
            <span>Llama-3 (Q4_K_M) + ChromaDB</span>
          </div>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={() =>
            setMessages([
              {
                role: "assistant",
                content:
                  "New conversation initiated. What are we building or querying today?",
                id: null,
              },
            ])
          }
        >
          <span>＋</span> New Session
        </button>

        <nav className="sidebar-nav">
          <button
            type="button"
            className={`nav-item ${!showConverter ? "active" : ""}`}
            onClick={() => setShowConverter(false)}
          >
            <span>💬</span> Workspace Chat
          </button>
          <button
            type="button"
            className={`nav-item ${showConverter ? "active" : ""}`}
            onClick={() => setShowConverter((prev) => !prev)}
          >
            <span>🔄</span> Format Converter
          </button>
        </nav>

        {/* Real-time Storage & Accuracy Metrics */}
        <div className="sidebar-section">
          <div className="section-title">
            <span>STORAGE OCCUPANCY</span>
            <span className="live-dot">● LIVE</span>
          </div>

          <div className="system-card">
            <div className="system-row">
              <span>Total DB Space</span>
              <strong>
                {metrics ? `${metrics.storage.total_occupied_mb} MB` : "--"}
              </strong>
            </div>
            <div className="system-row">
              <span>SQLite Metadata</span>
              <strong>
                {metrics ? `${metrics.storage.sqlite_mb} MB` : "--"}
              </strong>
            </div>
            <div className="system-row">
              <span>ChromaDB Vectors</span>
              <strong>
                {metrics ? `${metrics.storage.chroma_mb} MB` : "--"}
              </strong>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="section-title">
            <span>RAG PERFORMANCE</span>
          </div>

          <div className="metric">
            <span>Retrieval Sim</span>
            <strong className="cyan">
              {metrics ? `${metrics.performance.avg_retrieval_similarity}%` : "--"}
            </strong>
          </div>

          <div className="metric">
            <span>Grounding Score</span>
            <strong className="green">
              {metrics ? `${metrics.performance.avg_grounding_score}%` : "--"}
            </strong>
          </div>

          <div className="metric">
            <span>Avg CPU Latency</span>
            <strong>
              {metrics ? `${metrics.performance.avg_latency_seconds}s` : "--"}
            </strong>
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="privacy-icon">🔒</span>
          <div>
            <strong>100% Offline & Private</strong>
            <small>Zero telemetry • Local CPU</small>
          </div>
        </div>
      </aside>

      {/* Main Workspace Area */}
      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <div>
              <h2>Local AI Engineering Workspace</h2>
              <div className="connection-status">
                <span className="status-dot"></span>
                <span>Active Local Host • Privacy Protected</span>
              </div>
            </div>
          </div>

          <div className="topbar-actions">
            <span className="model-pill">Ollama • Llama-3-8B</span>
            <button
              type="button"
              className="icon-button"
              title="Toggle File Converter"
              onClick={() => setShowConverter((prev) => !prev)}
            >
              🔄
            </button>
          </div>
        </header>

        {/* Floating File Converter Panel */}
        {showConverter && (
          <div className="converter-panel">
            <div className="converter-title">
              <div>
                <span className="eyebrow">DOCUMENT UTILITY</span>
                <h3>Format Conversion Engine</h3>
              </div>
              <button
                type="button"
                className="close-button"
                onClick={() => setShowConverter(false)}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleConvertSubmit} className="converter-form">
              <input
                type="file"
                accept=".pdf,.txt,.docx,.pptx,.xlsx,.csv,.png,.jpg,.jpeg"
                onChange={(e) => {
                  setConvertFile(e.target.files?.[0] || null);
                  setConvertedUrl(null);
                  setConvertedFileName(null);
                }}
              />
              <select
                value={targetFormat}
                onChange={(e) => setTargetFormat(e.target.value)}
              >
                <option value="docx">Word (.docx)</option>
                <option value="pdf">PDF (.pdf)</option>
                <option value="xlsx">Excel (.xlsx)</option>
                <option value="csv">CSV (.csv)</option>
                <option value="pptx">PowerPoint (.pptx)</option>
              </select>
              <button type="submit" disabled={isConverting || !convertFile}>
                {isConverting ? "Processing..." : "Convert & Export"}
              </button>
            </form>

            {convertedUrl && (
              <div className="download-box">
                <span>{convertedFileName} ready!</span>
                <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                  <button
                    type="button"
                    onClick={() => triggerBrowserDownload(convertedUrl, convertedFileName)}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--cyan)",
                      cursor: "pointer",
                      padding: 0,
                      textDecoration: "underline",
                      font: "inherit",
                    }}
                  >
                    ⬇️ Download File Again
                  </button>
                  <a href={convertedUrl} target="_blank" rel="noreferrer">
                    (Direct Link)
                  </a>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Scrollable Chat Area */}
        <section className="chat-area">
          <div className="messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`message-row ${m.role}`}>
                <div
                  className={`message-avatar ${
                    m.role === "assistant" ? "ai" : "user-avatar"
                  }`}
                >
                  {m.role === "assistant" ? "⚡" : "👤"}
                </div>
                <div className="message-content">
                  <div className="message-name">
                    {m.role === "assistant" ? "ENGINEERING AGENT" : "USER"}
                  </div>
                  <div className="message-bubble">
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || "");
                          const codeString = String(children).replace(/\n$/, "");
                          return !inline && match ? (
                            <div className="code-container">
                              <div className="code-header">
                                <span>{match[1].toUpperCase()}</span>
                                <button
                                  type="button"
                                  onClick={() => copyCode(codeString)}
                                >
                                  Copy
                                </button>
                              </div>
                              <SyntaxHighlighter
                                style={atomDark}
                                language={match[1]}
                                PreTag="div"
                                {...props}
                              >
                                {codeString}
                              </SyntaxHighlighter>
                            </div>
                          ) : (
                            <code className="inline-code" {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                  </div>

                  {m.role === "assistant" && (
                    <div className="message-tools">
                      {m.id && (
                        <>
                          <button
                            type="button"
                            title="Helpful"
                            onClick={() => handleFeedback(m.id, 1)}
                          >
                            👍
                          </button>
                          <button
                            type="button"
                            title="Calibrate"
                            onClick={() => handleFeedback(m.id, -1)}
                          >
                            👎
                          </button>
                        </>
                      )}
                      {m.accuracy !== undefined && (
                        <span className="quality-tag">
                          Sim: {m.accuracy}% · Ground: {m.grounding}%
                        </span>
                      )}
                      {m.tokenSpeed !== undefined && (
                        <span className="quality-tag">
                          ⚡ {m.tokenSpeed} tok/s
                        </span>
                      )}
                      {m.modeUsed && (
                        <span className="mode-pill-tag">
                          {m.modeUsed === "chroma_only" ? "Docs" : m.modeUsed === "hybrid" ? "Hybrid" : "Model"}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && messages[messages.length - 1]?.content === "" && (
              <div className="message-row assistant">
                <div className="message-avatar ai">⚡</div>
                <div className="message-content">
                  <div className="message-name">ENGINEERING AGENT</div>
                  <div className="typing-message">
                    <span>
                      {engineMode === "chroma_only"
                        ? "Querying vector index & assembling documents..."
                        : engineMode === "model_only"
                        ? "Generating parametric response on local CPU..."
                        : "Evaluating hybrid context and local memory..."}
                    </span>
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </section>

        {/* Fixed Bottom Composer Area with Mode & Depth Bar */}
        <footer className="composer-area">
          <div className="composer-wrapper">
            {/* Quick Engine & Depth Bar */}
            <div className="controls-bar">
              <div className="control-group">
                <span className="control-label">SOURCE</span>
                <div className="pill-selector">
                  <button
                    type="button"
                    className={`pill-btn ${engineMode === "chroma_only" ? "active" : ""}`}
                    onClick={() => setEngineMode("chroma_only")}
                    title="Strict Document RAG (Zero Hallucination)"
                  >
                    Docs Only
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${engineMode === "hybrid" ? "active" : ""}`}
                    onClick={() => setEngineMode("hybrid")}
                    title="Context Grounded with General AI Fallback"
                  >
                    Hybrid
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${engineMode === "model_only" ? "active" : ""}`}
                    onClick={() => setEngineMode("model_only")}
                    title="Direct Llama 3 Parametric Generation"
                  >
                    Model Only
                  </button>
                </div>
              </div>

              <div className="control-group">
                <span className="control-label">DEPTH</span>
                <div className="pill-selector">
                  <button
                    type="button"
                    className={`pill-btn ${detailLevel === "short" ? "active-green" : ""}`}
                    onClick={() => setDetailLevel("short")}
                    title="Fast 2-4 sentence summary"
                  >
                    Short
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${detailLevel === "detailed" ? "active-green" : ""}`}
                    onClick={() => setDetailLevel("detailed")}
                    title="Structured technical breakdown"
                  >
                    Detailed
                  </button>
                  <button
                    type="button"
                    className={`pill-btn ${detailLevel === "full" ? "active-green" : ""}`}
                    onClick={() => setDetailLevel("full")}
                    title="Complete sequenced document extraction / exhaustive response"
                  >
                    Full Audit
                  </button>
                </div>
              </div>
            </div>

            <form onSubmit={handleSend} className="composer">
              <button
                type="button"
                className="attach-button"
                title="Ingest File (PDF, Office, Images with OCR)"
                onClick={() => fileInputRef.current?.click()}
              >
                📎
              </button>
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  engineMode === "chroma_only"
                    ? "Strictly query indexed documents..."
                    : engineMode === "model_only"
                    ? "Query Llama 3 internal weights directly..."
                    : "Ask a technical question, query your documents, or trigger tasks..."
                }
                disabled={loading}
              />
              <button
                type="submit"
                className="send-button"
                disabled={loading || !input.trim()}
                title="Send Message"
              >
                ↑
              </button>
            </form>
          </div>
          <div className="composer-hint">
            <span>
              Enter to send · Shift + Enter for new line · Mode: <strong>{engineMode.toUpperCase()}</strong> · Depth: <strong>{detailLevel.toUpperCase()}</strong>
            </span>
          </div>
        </footer>
      </main>
    </div>
  );
}