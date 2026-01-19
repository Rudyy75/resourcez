# 📚 Phase 4 Notes (Part 2): Days 27-30
## Docker, Visualization, Report, and Final Polish

> **Duration:** ~10-12 hours total  
> **Goal:** Containerize project, generate all visualizations, write report, polish for portfolio  
> **Output:** `Dockerfile`, `visualization.py`, `docs/final_report.md`, professional README

---

# 🗺️ Learning Roadmap

```
Day 27: Docker Container
├── Docker fundamentals
├── Dockerfile creation
├── Multi-stage builds

Day 28: Visualization Suite
├── Equity curve plots
├── Model comparison charts
├── Attention heatmaps

Day 29: Final Report
├── Research paper structure
├── Results presentation
├── Limitations discussion

Day 30: GitHub Polish
├── Professional README
├── Badges and shields
├── Pre-commit hooks
```

---

# 📺 Videos to Watch

| Order | Topic | Video | Duration | Priority |
|-------|-------|-------|----------|----------|
| 1 | Docker Quick | [Docker in 100 Seconds](https://youtu.be/Gjnup-PuquQ) | 2 min | 🔴 Must |
| 2 | Docker Python | [Docker for Python](https://youtu.be/0TFWtfFY87U) | 25 min | 🔴 Must |
| 3 | Matplotlib | [Matplotlib Tips](https://youtu.be/UO98lJQ3QGI) | 30 min | 🟡 Good |
| 4 | README Guide | [Write Good README](https://youtu.be/E6NO0rgFub4) | 8 min | 🔴 Must |

---

# Part 1: Docker Fundamentals (Day 27)

## 1.1 What is Docker?

```
┌──────────────────────────────────────────────────────────────┐
│                    WITHOUT DOCKER                             │
│                                                               │
│  Developer A: "Works on my machine!" (Python 3.8, Ubuntu)    │
│  Developer B: "Not on mine!" (Python 3.10, Windows)          │
│  Production: "Broken!" (Python 3.7, Alpine)                  │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│                     WITH DOCKER                               │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │           CONTAINER (Always identical)               │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐               │     │
│  │  │ Python  │ │  Your   │ │  All    │               │     │
│  │  │  3.10   │ │  Code   │ │  Deps   │               │     │
│  │  └─────────┘ └─────────┘ └─────────┘               │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Runs EXACTLY the same everywhere! ✓                         │
└──────────────────────────────────────────────────────────────┘
```

## 1.2 Docker Concepts

```
IMAGE vs CONTAINER

┌───────────────────────────┐
│         IMAGE             │    ← Like a RECIPE (read-only)
│  (Blueprint/Template)     │       or a SNAPSHOT
└───────────────────────────┘
            │
            │  docker run
            ▼
┌───────────────────────────┐
│        CONTAINER          │    ← Like a RUNNING PROGRAM
│   (Running Instance)      │       (can have many from 1 image)
└───────────────────────────┘

You can have MULTIPLE containers from ONE image
```

## 1.3 Dockerfile Explained

```dockerfile
# Dockerfile

# Base image - start from official Python
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create directories if needed
RUN mkdir -p data/raw data/processed models outputs/figures outputs/results

# Default command when container runs
CMD ["python", "main.py", "--help"]
```

### Line-by-Line Explanation

| Line | Purpose |
|------|---------|
| `FROM python:3.10-slim` | Use Python 3.10 on minimal Debian |
| `WORKDIR /app` | All commands run in /app |
| `COPY requirements.txt .` | Copy deps file first (cache layer) |
| `RUN pip install...` | Install all dependencies |
| `COPY . .` | Copy entire project |
| `CMD ["python"...]` | Default startup command |

## 1.4 Docker Commands

```bash
# Build image from Dockerfile
docker build -t market-oracle .

# Run container interactively
docker run -it market-oracle

# Run with custom command
docker run market-oracle python main.py --ticker AAPL

# Run with volume (share files with host)
docker run -v $(pwd)/data:/app/data market-oracle

# List running containers
docker ps

# Stop container
docker stop <container_id>

# Clean up old images
docker system prune
```

## 1.5 Docker Compose (Optional)

```yaml
# docker-compose.yml

version: '3.8'

services:
  market-oracle:
    build: .
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - NEWS_API_KEY=${NEWS_API_KEY}
    command: python main.py --ticker AAPL
```

```bash
# Run with docker-compose
docker-compose up
```

---

# Part 2: Visualization Suite (Day 28)

## 2.1 Essential Charts

```python
# visualization.py
"""
Market Oracle - Visualization Module

Generate all charts for the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")
FIGURE_DIR = Path("outputs/figures")
```

## 2.2 Equity Curve Plot

```python
def plot_equity_curve(
    strategy_equity: np.ndarray,
    benchmark_equity: np.ndarray,
    dates: pd.DatetimeIndex,
    save_path: str = None
):
    """
    Plot strategy vs benchmark equity curves.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, strategy_equity, label='ML Strategy', linewidth=2)
    ax.plot(dates, benchmark_equity, label='Buy & Hold', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Strategy Performance vs Buy & Hold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

## 2.3 Model Comparison Radar Chart

```python
def plot_model_comparison_radar(metrics_df: pd.DataFrame, save_path: str = None):
    """
    Radar chart comparing different models across metrics.
    """
    categories = ['Direction Acc', 'Sharpe', '1/MaxDD', 'CAGR', 'Win Rate']
    N = len(categories)
    
    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for idx, (model_name, row) in enumerate(metrics_df.iterrows()):
        values = [row['dir_acc'], row['sharpe'], 1/row['max_dd'], row['cagr'], row['win_rate']]
        # Normalize to 0-1
        values = [v / max(metrics_df[col]) for v, col in zip(values, metrics_df.columns)]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

## 2.4 Drawdown Chart

```python
def plot_drawdown(equity_curve: np.ndarray, dates: pd.DatetimeIndex, save_path: str = None):
    """
    Plot drawdown over time.
    """
    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max * 100
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(dates, drawdown, color='crimson', alpha=0.4)
    ax.plot(dates, drawdown, color='darkred', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Drawdown goes down
    ax.grid(True, alpha=0.3)
    
    # Mark max drawdown
    max_dd_idx = np.argmax(drawdown)
    ax.scatter(dates[max_dd_idx], drawdown[max_dd_idx], color='black', s=100, zorder=5)
    ax.annotate(f'Max: {drawdown[max_dd_idx]:.1f}%', 
                xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

## 2.5 Feature Importance Chart

```python
def plot_feature_importance(importance: pd.Series, save_path: str = None):
    """
    Horizontal bar chart of feature importance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    importance_sorted = importance.sort_values()
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_sorted)))
    
    bars = ax.barh(importance_sorted.index, importance_sorted.values, color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, importance_sorted.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

# Part 3: Final Report (Day 29)

## 3.1 Report Structure

```markdown
# docs/final_report.md

# Market Oracle: Stock Direction Prediction with LSTM and Attention

## Abstract (150 words)
This project implements a machine learning pipeline for predicting stock 
market direction using LSTM networks enhanced with attention mechanisms 
and sentiment analysis. We employ walk-forward validation to prevent 
data leakage and evaluate against naive baselines...

## 1. Introduction
### 1.1 Problem Statement
- Stock prediction is challenging
- Most retail traders lose money
- Can ML provide an edge?

### 1.2 Objectives
- Build production-ready ML pipeline
- Compare classical ML vs deep learning
- Measure impact of sentiment and attention
- Honest evaluation with transaction costs

## 2. Data & Preprocessing
### 2.1 Data Sources
- Price data: Yahoo Finance (yfinance)
- News data: NewsAPI
- Date range: 2015-2024

### 2.2 Features
| Feature | Description | Source |
|---------|-------------|--------|
| Log Return | log(P_t/P_{t-1}) | Price |
| RSI | Momentum oscillator | Price |
| MACD | Trend indicator | Price |
| Sentiment | VADER score | News |

### 2.3 Data Leakage Prevention
- Strict walk-forward validation
- Sentiment(t) → Predict(t+1)
- Scaling fit on training only

## 3. Methodology
### 3.1 Walk-Forward Validation
[Diagram of expanding window approach]

### 3.2 Model Architectures
#### Phase 1: Classical ML
- Logistic Regression
- Random Forest

#### Phase 2: LSTM
[Architecture diagram]

#### Phase 3: Attention + Sentiment
[Multi-input architecture diagram]

## 4. Results
### 4.1 Ablation Study
| Model | Dir Acc | RMSE | p-value |
|-------|---------|------|---------|
| Baseline | 52.0% | - | - |
| LSTM | 54.8% | 0.020 | 0.02* |
| Full Model | 56.2% | 0.018 | 0.005** |

### 4.2 Backtest Performance
[Equity curve chart]

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| CAGR | 12.5% | 8.2% |
| Sharpe | 1.24 | 0.85 |
| Max DD | 18.5% | 33.2% |

### 4.3 Multi-Ticker Generalization
[Generalization matrix]

## 5. Discussion
### 5.1 Key Findings
- Attention adds +1-2% accuracy
- Sentiment helps during high-news periods
- Model generalizes across tech stocks

### 5.2 Limitations
- Trained on bull market period
- News API has limited history
- No look-ahead bias in results

### 5.3 Future Work
- Transformer architecture
- Alternative data sources
- Real-time predictions

## 6. Conclusion
[Summary of key achievements]

## References
1. [Attention Is All You Need, Vaswani et al.]
2. [VADER Sentiment, Hutto & Gilbert]
...

## Appendix
### A. Code Structure
### B. Hyperparameters
### C. Reproducibility
```

---

# Part 4: GitHub Polish (Day 30)

## 4.1 Professional README Template

```markdown
# Market Oracle 🔮

> **ML-powered stock market direction prediction with LSTM, Attention, and Sentiment Analysis**

[![CI](https://github.com/USERNAME/Market-Oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/Market-Oracle/actions)
[![codecov](https://codecov.io/gh/USERNAME/Market-Oracle/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/Market-Oracle)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/)

---

## 📊 Results Summary

| Model | Direction Accuracy | Sharpe Ratio | vs Baseline |
|-------|-------------------|--------------|-------------|
| LSTM + Attention + Sentiment | **56.2%** | **1.24** | +4.2% |

![Equity Curve](outputs/figures/equity_curve.png)

---

## 🏗️ Architecture

[Mermaid diagram of full architecture]

---

## 🚀 Quick Start

### Option 1: Local Installation
```bash
git clone https://github.com/USERNAME/Market-Oracle.git
cd Market-Oracle
pip install -r requirements.txt
python main.py --ticker AAPL
```

### Option 2: Docker
```bash
docker build -t market-oracle .
docker run market-oracle python main.py --ticker AAPL
```

---

## 📁 Project Structure
[Directory tree]

---

## 📈 Key Features
- ✅ Walk-forward validation (no data leakage)
- ✅ Ablation study with statistical significance
- ✅ Attention mechanism visualization
- ✅ Full backtesting with transaction costs
- ✅ CI/CD with GitHub Actions
- ✅ Docker containerization

---

## 📝 License
MIT License - see [LICENSE](LICENSE)
```

## 4.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.292
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"What is Docker and why would you use it?"**
   - Packages app with all dependencies
   - Runs identically everywhere
   - Easy deployment and scaling

2. **"What's the difference between a Dockerfile and docker-compose?"**
   - Dockerfile: Builds single image
   - docker-compose: Orchestrates multiple containers
   - Use compose for complex setups

3. **"What are pre-commit hooks?"**
   - Run checks before git commit
   - Enforce code style automatically
   - Catch issues early

4. **"How do you structure a good README?"**
   - Clear title and description
   - Installation instructions
   - Usage examples
   - Results/demos
   - Contributing guidelines

## ML-Focused Questions

1. **"How would you present ML results to a non-technical audience?"**
   - Visual comparisons (equity curves)
   - Simple metrics (% improvement)
   - Real-world implications (profit/loss)

2. **"What limitations should you mention in an ML report?"**
   - Training data period
   - Assumptions made
   - Edge cases not handled
   - Computational requirements

3. **"How do you ensure reproducibility?"**
   - Random seeds documented
   - Exact package versions
   - Docker container
   - Data versioning

---

# ✅ Day 27-30 Checklist

## Day 27 Deliverables
- [ ] Watch Docker videos
- [ ] Create `Dockerfile`
- [ ] Build image: `docker build -t market-oracle .`
- [ ] Test container runs correctly
- [ ] Add Docker instructions to README

## Day 28 Deliverables
- [ ] Create `visualization.py`
- [ ] Generate equity curve plot
- [ ] Generate drawdown chart
- [ ] Generate feature importance chart
- [ ] Generate model comparison radar
- [ ] Save all plots to `outputs/figures/`

## Day 29 Deliverables
- [ ] Create `docs/final_report.md`
- [ ] Write all sections (abstract → conclusion)
- [ ] Include all charts and tables
- [ ] List limitations honestly
- [ ] Add references

## Day 30 Deliverables
- [ ] Polish README with badges
- [ ] Add results summary table
- [ ] Include architecture diagram
- [ ] Create `.pre-commit-config.yaml`
- [ ] Run pre-commit on all files
- [ ] Final test: `pytest tests/ -v`
- [ ] Final push to GitHub
- [ ] Verify CI/CD pipeline is green ✅

---

# 📖 Quick Reference

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

```bash
# Docker commands
docker build -t market-oracle .
docker run market-oracle
docker run -v $(pwd)/data:/app/data market-oracle

# Pre-commit
pre-commit install
pre-commit run --all-files
```

---

# 🏆 Congratulations!

You've completed all 30 days! Your project now has:

✅ Full ML pipeline (data → features → models → backtest)
✅ 7+ test files with pytest
✅ Type hints and mypy checking
✅ CI/CD with GitHub Actions
✅ Docker containerization
✅ Ablation study with significance tests
✅ Multi-ticker generalization
✅ Professional documentation

**This is a portfolio-ready, interview-worthy project!** 🚀
