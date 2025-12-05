// src/App.js
import React, { useState, useEffect } from "react";

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");

  const [prediction, setPrediction] = useState(null);
  const [currentView, setCurrentView] = useState("original"); // original | heatmap | overlay
  const [history, setHistory] = useState([]);

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("prediction_history");
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem("prediction_history", JSON.stringify(history));
  }, [history]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Basic size check: 5 MB
    if (file.size > 5 * 1024 * 1024) {
      setError("File size should be less than 5 MB");
      return;
    }

    setError("");
    setSelectedFile(file);
    setPrediction(null);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleUploadClick = () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }
    uploadImage();
  };

  const uploadImage = async () => {
    try {
      setIsUploading(true);
      setError("");

      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to get prediction");
      }

      const data = await res.json();

      const predObj = {
        predicted_class: data.predicted_class,
        predicted_index: data.predicted_index,
        confidence: data.confidence,
        probabilities: data.probabilities,
        gradcam_image: data.gradcam_image,
        original_image: previewUrl,
        timestamp: new Date().toISOString(),
      };

      setPrediction(predObj);
      setCurrentView("original");

      setHistory((prev) => [
        {
          id: Date.now(),
          thumbnail: previewUrl,
          prediction: data.predicted_class,
          confidence: data.confidence,
          timestamp: new Date().toLocaleString(),
        },
        ...prev.slice(0, 9), // keep max 10
      ]);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setIsUploading(false);
    }
  };

  const getPredictionLabel = () => {
    if (!prediction) return "";
    // Adjust this mapping based on your CLASS_NAMES
    // Example: ["FAKE", "REAL"]
    const label = prediction.predicted_class || "";
    return label.toUpperCase();
  };

  const getPredictionColorClass = () => {
    const label = getPredictionLabel();
    if (label.includes("REAL")) return "status-real";
    if (label.includes("FAKE") || label.includes("AI")) return "status-fake";
    return "status-neutral";
  };

  const formatConfidence = (val) => {
    if (!val && val !== 0) return "-";
    return `${(val * 100).toFixed(1)}%`;
  };


  return (
    <div className="app-root">
      <div className="page-container">
        {/* HEADER */}
        <header className="header">
          <h1>CIFAKE - AI Generated Image Detection System</h1>
          <p className="subtitle">
            Detect whether an image is AI-generated or real in seconds.
          </p>
        </header>

        {/* UPLOAD CARD */}
        <section className="card upload-card">
          <div className="upload-inner">
            <div className="upload-icon">⬆</div>
            <p className="upload-text">
              Upload an image (JPG / PNG • up to 5 MB)
            </p>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <button
              className="btn-primary"
              onClick={() => document.getElementById("file-input").click()}
            >
              Choose Image
            </button>

            {selectedFile && (
              <p className="file-name">Selected: {selectedFile.name}</p>
            )}

            <button
              className="btn-upload"
              onClick={handleUploadClick}
              disabled={isUploading || !selectedFile}
            >
              {isUploading ? "Analyzing..." : "Upload & Analyze"}
            </button>

            {error && <p className="error-text">{error}</p>}
          </div>
        </section>

        {/* PREDICTION + IMAGES */}
        {prediction && (
          <>
            {/* STATUS BAR */}
            <section className="card status-card">
              <div className={`status-pill ${getPredictionColorClass()}`}>
                {getPredictionLabel() || "PREDICTION"}
              </div>
              <div className="confidence-bar-wrapper">
                <div className="confidence-header">
                  <span>Confidence</span>
                  <span className="confidence-value">
                    {formatConfidence(prediction.confidence)}
                  </span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${(prediction.confidence || 0) * 100}%`,
                    }}
                  ></div>
                </div>
              </div>
            </section>

            {/* IMAGE VIEWER + TABS */}
            <section className="card image-section">
              <div className="tabs">
                <button
                  className={
                    currentView === "original"
                      ? "tab active-tab"
                      : "tab"
                  }
                  onClick={() => setCurrentView("original")}
                >
                  Original View
                </button>
                <button
                  className={
                    currentView === "heatmap" ? "tab active-tab" : "tab"
                  }
                  onClick={() => setCurrentView("heatmap")}
                >
                  Heatmap View
                </button>
                <button
                  className={
                    currentView === "overlay" ? "tab active-tab" : "tab"
                  }
                  onClick={() => setCurrentView("overlay")}
                >
                  Overlay View
                </button>
              </div>

              <div className="image-viewer">
                <div className="image-columns">
                  <div className="image-column">
                    <h4>Original Image</h4>
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Original preview"
                        className="image-display"
                      />
                    ) : (
                      <div className="image-placeholder">
                        No image selected
                      </div>
                    )}
                  </div>
                  <div className="image-column">
                    <h4>Grad-CAM Heatmap</h4>
                    {prediction.gradcam_image ? (
                      <img
                        src={prediction.gradcam_image}
                        alt="Grad-CAM heatmap"
                        className="image-display"
                      />
                    ) : (
                      <div className="image-placeholder">
                        No Grad-CAM available
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>

            {/* METRICS */}
            <section className="card metrics-section">
              <h3>Model Performance Metrics</h3>
              <div className="metrics-grid">
                <MetricCard label="Accuracy" value="93.12%" />
                <MetricCard label="Precision" value="92.75%" />
                <MetricCard label="Recall" value="93.42%" />
                <MetricCard label="F1-score" value="93.06%" />
              </div>
            </section>
          </>
        )}

        {/* HISTORY */}
        <section className="card history-section">
          <div className="history-header">
            <h3>Recent History</h3>
            <button
              className="btn-clear"
              onClick={() => setHistory([])}
              disabled={history.length === 0}
            >
              Clear
            </button>
          </div>
          {history.length === 0 ? (
            <p className="muted-text">
              No predictions yet. Upload an image to get started.
            </p>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Thumbnail</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <img
                          src={item.thumbnail}
                          alt="thumb"
                          className="thumb-img"
                        />
                      </td>
                      <td>
                        <span
                          className={
                            item.prediction.toUpperCase().includes("REAL")
                              ? "badge badge-real"
                              : "badge badge-fake"
                          }
                        >
                          {item.prediction.toUpperCase()}
                        </span>
                      </td>
                      <td>{formatConfidence(item.confidence)}</td>
                      <td>{item.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* FOOTER */}
        <footer className="footer">
          <span>© 2025 Team StoryMinds. All rights reserved.</span>
          <div className="footer-links">
            <a href="#about">About the Model</a>
            <a href="#repo">GitHub Repo</a>
            <span>Team StoryMinds</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-bar">
        <div className="metric-bar-fill"></div>
      </div>
    </div>
  );
}

export default App;
