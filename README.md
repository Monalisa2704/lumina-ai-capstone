# 🚀 Lumina AI: Autonomous Creative Director
**A Multi-Agent System for Data-Driven E-Commerce Marketing**

Lumina AI is an Agent-to-Agent (A2A) pipeline built for the Kaggle 5-Day AI Agents Intensive. It connects visual product asset processing with generative copy creation and predictive performance scoring.

---

## 🛠️ System Architecture

* **Worker 1 (The Editor Agent):** Extracts and isolates product assets from raw photos using `rembg`, then loads them onto an interactive HTML5 canvas for styling.
* **Worker 2 (The Marketer Agent):** Runs under Model Context Protocol (MCP) brand guidelines and uses Gemini 2.5 Flash with live Google Search tool calling to write trend-aware copy.
* **Worker 3 (The Evaluator Agent):** Uses a Scikit-Learn Random Forest Regressor trained on campaign performance features to score engagement rates and rank options.

---

## 📂 Repository Structure

```text
├── templates/
│   ├── index.html          # Studio interface for asset editing
│   └── marketer.html       # Marketing desk for copy generation
├── static/
│   └── uploads/            # Asset storage
├── app.py                  # Core Flask server & agent routing
├── generate_dataset.py     # Synthetic training data generator
├── marketing_campaigns.csv # 5,000-row feature dataset
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
