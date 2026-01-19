# 📚 Phase 3 Notes (Part 2): Days 19-22
## MLflow, Ablation Studies, and Multi-Ticker Validation

> **Duration:** ~10-12 hours total  
> **Goal:** Track experiments properly, prove each component adds value  
> **Output:** MLflow tracking, ablation notebook, generalization results

---

# 🗺️ Learning Roadmap

```
Day 19: MLflow Setup
├── Why experiment tracking
├── MLflow basics
├── Logging metrics and artifacts

Day 20: Ablation Studies
├── What is ablation
├── Statistical significance
├── Building ablation table

Day 21: Multi-Ticker Validation
├── Testing on 5 tickers
├── Generalization analysis
├── Failure case identification

Day 22: Phase 3 Documentation
├── Attention heatmaps
├── Sentiment impact analysis
└── README updates
```

---

# 📺 Videos to Watch

| Order | Topic | Video | Duration | Priority |
|-------|-------|-------|----------|----------|
| 1 | MLflow Intro | [MLflow in 15 minutes](https://youtu.be/x3cxvsUFVZA) | 15 min | 🔴 Must |
| 2 | Ablation Studies | [What is Ablation?](https://towardsdatascience.com/what-is-ablation-study-in-machine-learning-5f3f4d0c0) | 10 min read | 🔴 Must |
| 3 | Statistical Tests | [StatQuest - t-test](https://youtu.be/5koKb5B_YWo) | 12 min | 🟡 Good |
| 4 | Overfitting | [StatQuest - Overfitting](https://youtu.be/Q81RR3yKn30) | 9 min | 🟡 Good |

---

# Part 1: MLflow for Experiment Tracking (Day 19)

## 1.1 Why Track Experiments?

```
WITHOUT TRACKING:
─────────────────────────────────────────────────
Week 1: "LSTM with 64 units worked great!"
Week 2: "Changed something, now it's worse..."
Week 3: "What were my best settings again??"

WITH MLFLOW:
─────────────────────────────────────────────────
┌────────────────────────────────────────────────┐
│ Experiment: Market Oracle                       │
├──────┬──────────┬────────┬───────┬────────────┤
│ Run  │ LSTM     │ Dropout│ RMSE  │ Dir Acc    │
├──────┼──────────┼────────┼───────┼────────────┤
│ #1   │ 32       │ 0.1    │ 0.024 │ 52.3%      │
│ #2   │ 64       │ 0.2    │ 0.019 │ 55.1%  ⭐  │
│ #3   │ 128      │ 0.3    │ 0.021 │ 53.8%      │
└──────┴──────────┴────────┴───────┴────────────┘

Every experiment saved, compared, reproducible!
```

## 1.2 MLflow Concepts

```
MLflow Structure:
─────────────────────────────────────────────────

EXPERIMENT (e.g., "Market Oracle LSTM")
    │
    ├── RUN #1 (specific training session)
    │   ├── Parameters: {lstm_units: 64, dropout: 0.2}
    │   ├── Metrics: {rmse: 0.019, accuracy: 0.55}
    │   └── Artifacts: [model.h5, learning_curve.png]
    │
    ├── RUN #2
    │   ├── Parameters: {lstm_units: 128, dropout: 0.3}
    │   ├── Metrics: {rmse: 0.021, accuracy: 0.53}
    │   └── Artifacts: [model.h5, learning_curve.png]
    │
    └── RUN #3 ...
```

## 1.3 MLflow Basic Usage

```python
import mlflow
import mlflow.keras

# Set experiment name
mlflow.set_experiment("market_oracle_lstm")

# Start a run
with mlflow.start_run(run_name="lstm_baseline"):
    
    # Log hyperparameters
    mlflow.log_param("lstm_units", 64)
    mlflow.log_param("dropout", 0.2)
    mlflow.log_param("window_size", 30)
    mlflow.log_param("learning_rate", 0.001)
    
    # Train model
    history = model.fit(X_train, y_train, ...)
    
    # Log metrics (per fold or final)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("direction_accuracy", dir_acc)
    
    # Log metrics per fold
    for fold, metrics in enumerate(fold_results):
        mlflow.log_metric("rmse", metrics['rmse'], step=fold)
    
    # Log artifacts (files)
    mlflow.log_artifact("outputs/figures/learning_curve.png")
    
    # Log model
    mlflow.keras.log_model(model, "model")

# View results
# Run: mlflow ui
# Open: http://localhost:5000
```

## 1.4 MLflow Utility Module

```python
# utils/mlflow_utils.py
import mlflow
from typing import Dict, Any

def start_experiment(name: str) -> None:
    """Initialize MLflow experiment."""
    mlflow.set_experiment(name)

def log_params(params: Dict[str, Any]) -> None:
    """Log multiple parameters at once."""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_fold_metrics(fold: int, metrics: Dict[str, float]) -> None:
    """Log metrics for a specific fold."""
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value, step=fold)

def log_model_comparison(results_df) -> None:
    """Log comparison table as artifact."""
    results_df.to_csv("outputs/results/model_comparison.csv")
    mlflow.log_artifact("outputs/results/model_comparison.csv")
```

## 1.5 Running MLflow UI

```bash
# In terminal (from project directory)
mlflow ui

# Opens browser at http://localhost:5000
# Compare runs, view metrics, download artifacts
```

---

# Part 2: Ablation Studies (Day 20)

## 2.1 What is Ablation?

```
ABLATION = Systematically removing components to measure their impact

         Full Model: LSTM + Attention + Sentiment
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Remove Attention   Remove Sentiment   Remove Both
         │                 │                 │
         ▼                 ▼                 ▼
     Measure Δ        Measure Δ         Measure Δ
         │                 │                 │
         ▼                 ▼                 ▼
   "Attention adds    "Sentiment adds   "Together add
    +1.2% accuracy"    +0.8% accuracy"   +2.5% accuracy"
```

## 2.2 Why Ablation Matters

```
Interviewer: "Does the attention mechanism actually help?"

WITHOUT ABLATION:
  You: "Umm... I think so? The model does well..."
  Interviewer: 😐

WITH ABLATION:
  You: "Yes! I ran a controlled experiment. LSTM alone gets 54.2%
        direction accuracy. Adding attention improves to 55.4%.
        The difference is statistically significant (p=0.03).
        Here's my ablation table..."
  Interviewer: 🤩
```

## 2.3 Ablation Table Design

```python
# Your ablation study should produce this:

| Model Variant          | Dir Acc | RMSE   | Δ vs Baseline | p-value |
|------------------------|---------|--------|---------------|---------|
| Naive (Always UP)      | 52.0%   | -      | baseline      | -       |
| Logistic Regression    | 53.5%   | 0.024  | +1.5%         | 0.12    |
| Random Forest          | 54.2%   | 0.022  | +2.2%         | 0.05*   |
| LSTM only              | 54.8%   | 0.020  | +2.8%         | 0.02*   |
| LSTM + Attention       | 55.4%   | 0.019  | +3.4%         | 0.01*   |
| LSTM + Sentiment       | 55.1%   | 0.019  | +3.1%         | 0.02*   |
| LSTM + Attn + Sent     | 56.2%   | 0.018  | +4.2%         | 0.005** |

* = significant at p<0.05
** = significant at p<0.01
```

## 2.4 Implementing Statistical Significance

```python
from scipy import stats
import numpy as np

def paired_ttest(scores_a: list, scores_b: list) -> tuple:
    """
    Paired t-test comparing two models across the same folds.
    
    Args:
        scores_a: Accuracy scores for model A (per fold)
        scores_b: Accuracy scores for model B (per fold)
    
    Returns:
        t_statistic, p_value
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return t_stat, p_value

# Example usage
lstm_scores = [0.54, 0.55, 0.53, 0.56, 0.54]  # 5 folds
attention_scores = [0.55, 0.57, 0.54, 0.57, 0.56]

t_stat, p_value = paired_ttest(lstm_scores, attention_scores)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant! ✓")
else:
    print("Difference might be due to random chance")
```

## 2.5 Full Ablation Study Notebook

```python
# notebooks/03_ablation_study.ipynb

import pandas as pd
from scipy import stats

# Store results for all model variants
all_results = {
    'Naive': {'dir_acc': [], 'rmse': []},
    'LogReg': {'dir_acc': [], 'rmse': []},
    'RF': {'dir_acc': [], 'rmse': []},
    'LSTM': {'dir_acc': [], 'rmse': []},
    'LSTM+Attn': {'dir_acc': [], 'rmse': []},
    'LSTM+Sent': {'dir_acc': [], 'rmse': []},
    'Full': {'dir_acc': [], 'rmse': []}
}

# For each fold in walk-forward
for fold, (train_idx, test_idx) in enumerate(validator.split(data)):
    # ... prepare data ...
    
    # Model 1: Naive baseline
    naive_acc = (y_test > 0).mean()  # Accuracy of "always UP"
    all_results['Naive']['dir_acc'].append(naive_acc)
    
    # Model 2: Logistic Regression
    # ... train and evaluate ...
    all_results['LogReg']['dir_acc'].append(lr_acc)
    
    # Model 3: Random Forest
    # ... train and evaluate ...
    all_results['RF']['dir_acc'].append(rf_acc)
    
    # Model 4: LSTM only
    # ... train LSTM without attention ...
    all_results['LSTM']['dir_acc'].append(lstm_acc)
    
    # Model 5: LSTM + Attention
    # ... train LSTM with attention ...
    all_results['LSTM+Attn']['dir_acc'].append(attn_acc)
    
    # Model 6: LSTM + Sentiment (no attention)
    # ... train LSTM with sentiment features ...
    all_results['LSTM+Sent']['dir_acc'].append(sent_acc)
    
    # Model 7: Full model
    # ... train full model ...
    all_results['Full']['dir_acc'].append(full_acc)

# Create summary table
summary = []
baseline = all_results['Naive']['dir_acc']

for model_name, metrics in all_results.items():
    mean_acc = np.mean(metrics['dir_acc'])
    std_acc = np.std(metrics['dir_acc'])
    
    if model_name == 'Naive':
        delta = '-'
        p_val = '-'
    else:
        delta = mean_acc - np.mean(baseline)
        _, p_val = stats.ttest_rel(baseline, metrics['dir_acc'])
    
    summary.append({
        'Model': model_name,
        'Mean Dir Acc': f"{mean_acc:.1%}",
        'Std': f"±{std_acc:.1%}",
        'Δ vs Baseline': f"+{delta:.1%}" if delta != '-' else '-',
        'p-value': f"{p_val:.4f}" if p_val != '-' else '-'
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_markdown(index=False))
```

---

# Part 3: Multi-Ticker Generalization (Day 21)

## 3.1 Why Test Multiple Tickers?

```
OVERFITTING TO ONE TICKER:

Trained on AAPL only:
  AAPL: 57% accuracy ✓
  MSFT: 51% accuracy ✗
  JPM:  49% accuracy ✗

Model learned APPLE-specific patterns, not general stock patterns!

GOOD GENERALIZATION:

Trained on AAPL, tested on all:
  AAPL: 55% accuracy
  MSFT: 54% accuracy  
  JPM:  53% accuracy
  XOM:  52% accuracy
  AMZN: 54% accuracy

Consistent performance = generalizable model!
```

## 3.2 Ticker Diversity

```
Choose tickers from DIFFERENT SECTORS to test generalization:

| Ticker | Sector        | Why Include                          |
|--------|---------------|--------------------------------------|
| AAPL   | Technology    | High volume, news-driven             |
| MSFT   | Technology    | Similar to AAPL, correlation test    |
| JPM    | Finance       | Different dynamics, interest rates   |
| XOM    | Energy        | Commodity-driven, different patterns |
| AMZN   | Consumer      | Growth stock, earnings-sensitive     |
```

## 3.3 Multi-Ticker Validation Protocol

```python
def run_multi_ticker_validation(tickers: list, model_builder: callable):
    """
    Test model generalization across multiple tickers.
    
    Protocol:
    1. Train on each ticker separately
    2. Test on same ticker (in-sample)
    3. Test on other tickers (cross-sample)
    """
    results = []
    
    for train_ticker in tickers:
        # Train model on this ticker
        train_data = load_data(train_ticker)
        model = model_builder()
        model.fit(train_data)
        
        for test_ticker in tickers:
            # Test on all tickers
            test_data = load_data(test_ticker)
            accuracy = evaluate(model, test_data)
            
            results.append({
                'Train': train_ticker,
                'Test': test_ticker,
                'Accuracy': accuracy,
                'Type': 'In-sample' if train_ticker == test_ticker else 'Cross-sample'
            })
    
    return pd.DataFrame(results)
```

## 3.4 Generalization Matrix

```
                    TEST TICKER
                 AAPL   MSFT   JPM    XOM   AMZN
            ┌─────────────────────────────────────┐
       AAPL │ 55%*   54%    52%    51%   53%     │
TRAIN  MSFT │ 53%    56%*   51%    50%   52%     │
       JPM  │ 51%    50%    54%*   52%   51%     │
       XOM  │ 50%    49%    51%    55%*  50%     │
       AMZN │ 53%    52%    50%    51%   55%*    │
            └─────────────────────────────────────┘
            
* = In-sample (trained and tested on same ticker)

OBSERVATIONS:
- Diagonal (in-sample) is highest → expected!
- Same-sector transfer better (AAPL→MSFT: 54%)
- Cross-sector transfer harder (XOM→AMZN: 50%)
```

## 3.5 Failure Case Analysis

```python
def analyze_failures(y_true, y_pred, dates, threshold=0.1):
    """
    Identify and analyze model failure cases.
    """
    # Find large errors
    errors = np.abs(y_true - y_pred)
    large_errors = errors > threshold
    
    failure_dates = dates[large_errors]
    
    print("Dates with largest prediction errors:")
    for date in failure_dates[:10]:
        print(f"  {date}: Predicted {y_pred[date]:.3f}, Actual {y_true[date]:.3f}")
    
    # Analyze patterns
    failure_analysis = {
        'earnings_days': sum(is_earnings_day(d) for d in failure_dates),
        'high_volume_days': sum(is_high_volume(d) for d in failure_dates),
        'news_spike_days': sum(is_news_spike(d) for d in failure_dates),
    }
    
    return failure_analysis
```

---

# Part 4: Documentation & Visualization (Day 22)

## 4.1 Attention Heatmap Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention_weights, dates, n_samples=5):
    """
    Visualize attention weights across time.
    
    Args:
        attention_weights: Shape (n_samples, window_size)
        dates: Date labels for x-axis
        n_samples: Number of samples to show
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
    
    for i, ax in enumerate(axes):
        weights = attention_weights[i].flatten()
        ax.bar(range(len(weights)), weights, color='steelblue')
        ax.set_ylabel(f'Sample {i+1}')
        ax.set_ylim(0, max(weights) * 1.2)
        
        # Highlight top-3 attended days
        top_indices = np.argsort(weights)[-3:]
        for idx in top_indices:
            ax.bar(idx, weights[idx], color='orangered')
    
    axes[-1].set_xlabel('Days in Window')
    plt.tight_layout()
    plt.savefig('outputs/figures/attention_heatmap.png', dpi=150)
    plt.show()
```

## 4.2 Sentiment Impact Visualization

```python
def plot_sentiment_impact(df):
    """
    Show relationship between sentiment and next-day returns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter: Sentiment vs Returns
    ax1 = axes[0]
    ax1.scatter(df['sentiment_compound'], df['next_return'], alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.axvline(x=0, color='gray', linestyle='--')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Next Day Return')
    ax1.set_title('Sentiment vs Next-Day Return')
    
    # Binned: Average return by sentiment bucket
    ax2 = axes[1]
    df['sent_bucket'] = pd.cut(df['sentiment_compound'], 
                               bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                               labels=['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos'])
    avg_return = df.groupby('sent_bucket')['next_return'].mean()
    avg_return.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_ylabel('Average Next Day Return')
    ax2.set_title('Average Return by Sentiment Level')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/sentiment_impact.png', dpi=150)
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"Why use experiment tracking like MLflow?"**
   - Reproducibility: Know exact settings for any result
   - Comparison: Compare hundreds of experiments easily
   - Collaboration: Team can see all experiments
   - Deployment: Track which model is in production

2. **"How do you handle experiment versioning?"**
   - Each run has unique ID
   - Log git commit hash as parameter
   - Save code snapshot as artifact
   - Use MLflow model registry for production

3. **"What's the difference between parameters and metrics in MLflow?"**
   - Parameters: Input settings (lstm_units=64)
   - Metrics: Output measurements (accuracy=0.55)
   - Parameters are logged once, metrics can be logged per step

## ML-Focused Questions

1. **"What is an ablation study and why is it important?"**
   - Systematically removing components to measure impact
   - Proves each component contributes value
   - Required for research papers
   - Shows you understand your model

2. **"How do you determine if a result is statistically significant?"**
   - Run experiment multiple times (walk-forward folds)
   - Use paired t-test or Wilcoxon test
   - p < 0.05 typically means significant
   - Report confidence intervals

3. **"What does it mean if a model doesn't generalize across tickers?"**
   - Model overfitted to specific stock patterns
   - Features might be ticker-specific
   - Need more regularization or simpler model
   - Consider training on multiple tickers

4. **"How would you debug a model that works on AAPL but fails on JPM?"**
   - Compare feature distributions between tickers
   - Check if sectors have different dynamics
   - Analyze failure cases for patterns
   - Consider sector-specific models

5. **"What's the difference between validation and test sets in walk-forward?"**
   - Validation: Part of training fold for early stopping
   - Test: Held-out future data for evaluation
   - Never use test set for any model decisions

---

# ✅ Day 19-22 Checklist

## Day 19 Deliverables
- [ ] Watch MLflow tutorial
- [ ] Install MLflow: `pip install mlflow`
- [ ] Create `utils/mlflow_utils.py`
- [ ] Add MLflow tracking to training notebook
- [ ] Run `mlflow ui` and verify experiments appear

## Day 20 Deliverables
- [ ] Understand ablation study concept
- [ ] Create `notebooks/03_ablation_study.ipynb`
- [ ] Train all model variants (7 models)
- [ ] Compute statistical significance tests
- [ ] Generate ablation table with p-values

## Day 21 Deliverables
- [ ] Test on 5 diverse tickers
- [ ] Create generalization matrix
- [ ] Identify failure cases
- [ ] Document which sectors work best/worst

## Day 22 Deliverables
- [ ] Generate attention heatmaps
- [ ] Create sentiment impact plots
- [ ] Update README with Phase 3 results
- [ ] Run all tests: `pytest tests/ -v`
- [ ] Push to GitHub

---

# 📖 Quick Reference

```python
# MLflow
import mlflow
mlflow.set_experiment("market_oracle")
with mlflow.start_run():
    mlflow.log_param("lstm_units", 64)
    mlflow.log_metric("accuracy", 0.55)
    mlflow.log_artifact("plot.png")

# Statistical Significance
from scipy import stats
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
significant = p_value < 0.05

# Ablation Pattern
models = ['Baseline', 'Model-A', 'Model-B', 'Full']
for model in models:
    train_and_evaluate(model)
    compare_vs_baseline(model)

# Multi-Ticker Validation
for train_ticker in tickers:
    model = train(train_ticker)
    for test_ticker in tickers:
        evaluate(model, test_ticker)
```

---

Phase 3 complete! You now have a fully validated model with ablation studies! 🚀
