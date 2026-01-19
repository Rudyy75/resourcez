# 📚 Phase 4 Notes (Part 1): Days 23-26
## Backtesting Engine + Production Pipeline

> **Duration:** ~10-12 hours total  
> **Goal:** Build trading simulation, create CLI, set up CI/CD  
> **Output:** `backtester.py`, `main.py`, GitHub Actions workflow

---

# 🗺️ Learning Roadmap

```
Day 23: Backtesting Basics
├── Trading strategy logic
├── Transaction costs
├── Equity curve generation

Day 24: Performance Metrics
├── CAGR, Sharpe Ratio
├── Max Drawdown
├── Buy-and-hold comparison

Day 25: Pipeline Integration
├── End-to-end main.py
├── CLI with argparse/click
└── Config loading

Day 26: CI/CD Pipeline
├── GitHub Actions
├── pytest in CI
├── Code coverage
```

---

# 📺 Videos to Watch

| Order | Topic | Video | Duration | Priority |
|-------|-------|-------|----------|----------|
| 1 | Backtesting 101 | [QuantInsti Backtesting](https://youtu.be/vC7IJXz-s4E) | 30 min | 🔴 Must |
| 2 | Sharpe Ratio | [Sharpe Ratio Explained](https://youtu.be/kx3b8d-P4YY) | 8 min | 🔴 Must |
| 3 | CLI with argparse | [argparse Tutorial](https://youtu.be/cdblJqEUDNo) | 15 min | 🟡 Good |
| 4 | GitHub Actions | [GitHub Actions in 10 min](https://youtu.be/R8_veQiYBjI) | 10 min | 🔴 Must |
| 5 | Python CI | [Python CI Tutorial](https://youtu.be/mFFXuXjVgkU) | 20 min | 🟡 Good |

---

# Part 1: Backtesting Fundamentals (Day 23)

## 1.1 What is Backtesting?

```
BACKTESTING = Simulating a trading strategy on historical data

┌─────────────────────────────────────────────────────────────┐
│                     THE BACKTESTING LOOP                     │
│                                                              │
│   Day 1:                                                     │
│   ├── Model predicts: +0.3% return                          │
│   ├── Decision: predicted > 0? → BUY                        │
│   ├── Execute: Buy at close price                           │
│   └── Record position                                        │
│                                                              │
│   Day 2:                                                     │
│   ├── Model predicts: -0.1% return                          │
│   ├── Decision: predicted < 0? → SELL (go to cash)          │
│   ├── Execute: Sell at close price                          │
│   ├── Pay transaction cost: 10 bps                          │
│   └── Record new position                                    │
│                                                              │
│   ... repeat for all days ...                               │
│                                                              │
│   Final: Calculate total return, Sharpe, drawdown           │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 Simple Strategy Logic

```python
def generate_signals(predictions: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert predictions to trading signals.
    
    Args:
        predictions: Model's predicted returns
        threshold: Minimum predicted return to go long
    
    Returns:
        signals: 1 = Long (hold stock), 0 = Cash
    """
    return (predictions > threshold).astype(int)

# Example:
predictions = [0.01, -0.005, 0.02, -0.01, 0.005]
signals = generate_signals(predictions)
# signals = [1, 0, 1, 0, 1]  # Buy, Sell, Buy, Sell, Buy
```

## 1.3 Transaction Costs (CRITICAL!)

```
WHY TRANSACTION COSTS MATTER:

Without costs:
  100 trades × 0.5% avg profit = 50% total return! 🎉

With 10 bps (0.1%) cost per trade:
  100 trades × 0.5% profit = 50%
  100 trades × 0.1% cost = -10%
  Net return = 40%  😐

With 10 trades per year (more realistic):
  10 trades × 0.5% profit = 5%
  10 trades × 0.1% cost = -1%
  Net return = 4%  

TRANSACTION COSTS EAT YOUR PROFITS!
```

### Implementing Transaction Costs

```python
def apply_transaction_costs(
    returns: np.ndarray,
    signals: np.ndarray,
    cost_bps: float = 10
) -> np.ndarray:
    """
    Subtract transaction costs when position changes.
    
    Args:
        returns: Daily returns
        signals: Position signals (0 or 1)
        cost_bps: Cost in basis points (10 bps = 0.1%)
    
    Returns:
        returns_after_costs
    """
    cost = cost_bps / 10000  # Convert bps to decimal
    
    # Find where position changes
    position_changes = np.diff(signals, prepend=signals[0])
    trades = np.abs(position_changes)  # 1 where we traded
    
    # Subtract cost on trade days
    costs = trades * cost
    
    return returns - costs
```

## 1.4 Equity Curve

```python
def compute_equity_curve(
    returns: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 100000
) -> np.ndarray:
    """
    Compute portfolio value over time.
    
    Strategy return = stock return × position
    If position = 0 (cash), return = 0
    """
    # Strategy returns: only earn when holding
    strategy_returns = returns * signals
    
    # Cumulative return
    cumulative = (1 + strategy_returns).cumprod()
    
    # Equity = initial capital × cumulative return
    equity = initial_capital * cumulative
    
    return equity
```

### Visualization

```
EQUITY CURVE:

Value │               
150k  │                    ___/
      │             ___/--/
125k  │        ___/
      │   ___/
100k  │__/
      └─────────────────────────────
           Time (Days)
      
Compare YOUR strategy vs BUY-AND-HOLD baseline!
```

---

# Part 2: Performance Metrics (Day 24)

## 2.1 CAGR (Compound Annual Growth Rate)

```
CAGR = (Final_Value / Initial_Value)^(1/years) - 1

Example:
  Initial: $100,000
  Final: $150,000
  Years: 3
  
  CAGR = (150000/100000)^(1/3) - 1 = 14.5%
  
  "On average, strategy grew 14.5% per year"
```

```python
def compute_cagr(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        equity_curve: Portfolio values
        periods_per_year: Trading days per year (252)
    """
    total_return = equity_curve[-1] / equity_curve[0]
    years = len(equity_curve) / periods_per_year
    return total_return ** (1/years) - 1
```

## 2.2 Sharpe Ratio (MOST IMPORTANT!)

```
Sharpe = (Return - Risk_Free_Rate) / Volatility

In words: "Return per unit of risk"

Interpretation:
  Sharpe < 0:   Losing money
  Sharpe 0-1:   Poor strategy
  Sharpe 1-2:   Decent strategy ✓
  Sharpe 2-3:   Very good ✓✓
  Sharpe > 3:   Either amazing or data error 🤔
```

```python
def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Daily strategy returns
        risk_free_rate: Annual risk-free rate (2%)
    """
    # Annualized return
    annual_return = np.mean(returns) * 252
    
    # Annualized volatility
    annual_vol = np.std(returns) * np.sqrt(252)
    
    # Sharpe
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    return sharpe
```

## 2.3 Maximum Drawdown

```
DRAWDOWN = How much you lost from peak

         Peak
          │     
$120k ────●───────────
          │\
$100k ─────│\─────────  ← Trough (lowest point)
           │ \   __/    
$80k  ─────│──\-/─────
           │   │
           └───┘
         Drawdown = (120k - 80k)/120k = 33%
         
Maximum Drawdown = Worst drawdown ever experienced
```

```python
def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown (worst peak-to-trough decline).
    """
    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max
    
    return np.max(drawdowns)
```

## 2.4 Additional Metrics

```python
def compute_all_metrics(equity_curve: np.ndarray, returns: np.ndarray, signals: np.ndarray) -> dict:
    """Compute all backtest metrics."""
    
    # Find trades
    trades = np.abs(np.diff(signals, prepend=signals[0]))
    n_trades = trades.sum()
    
    # Winning trades
    trade_returns = returns[trades == 1]
    wins = (trade_returns > 0).sum()
    
    return {
        'Total Return': equity_curve[-1] / equity_curve[0] - 1,
        'CAGR': compute_cagr(equity_curve),
        'Sharpe Ratio': compute_sharpe(returns * signals),
        'Max Drawdown': compute_max_drawdown(equity_curve),
        'Win Rate': wins / n_trades if n_trades > 0 else 0,
        'Num Trades': n_trades,
        'Profit Factor': trade_returns[trade_returns > 0].sum() / abs(trade_returns[trade_returns < 0].sum())
    }
```

## 2.5 Comparison vs Buy-and-Hold

```python
def compare_with_benchmark(strategy_equity, actual_returns, initial_capital=100000):
    """
    Compare strategy vs buy-and-hold.
    """
    # Buy and hold equity
    bh_equity = initial_capital * (1 + actual_returns).cumprod()
    
    # Metrics for both
    strategy_metrics = compute_all_metrics(strategy_equity, ...)
    bh_metrics = compute_all_metrics(bh_equity, ...)
    
    print(f"{'Metric':<20} {'Strategy':>12} {'Buy & Hold':>12}")
    print("=" * 46)
    for metric in ['CAGR', 'Sharpe Ratio', 'Max Drawdown']:
        print(f"{metric:<20} {strategy_metrics[metric]:>12.2%} {bh_metrics[metric]:>12.2%}")
```

---

# Part 3: Pipeline Integration (Day 25)

## 3.1 Main Pipeline Script

```python
# main.py
"""
Market Oracle - End-to-end ML trading pipeline.

Usage:
    python main.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
"""

import argparse
import yaml
from pathlib import Path

from data_loader import download_ticker_data, clean_data, compute_log_returns
from indicators import add_all_indicators
from windowing import create_windows
from walk_forward import WalkForwardValidator
from models.attention_lstm import build_attention_lstm_model
from backtester import Backtester
from utils.logger import logger

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(ticker: str, start_date: str, end_date: str, config: dict):
    """Run full pipeline."""
    
    logger.info(f"Starting Market Oracle for {ticker}")
    
    # Step 1: Load and clean data
    logger.info("Step 1: Loading data...")
    df = download_ticker_data(ticker, start_date, end_date)
    df = clean_data(df)
    df['log_return'] = compute_log_returns(df)
    
    # Step 2: Add indicators
    logger.info("Step 2: Computing indicators...")
    df = add_all_indicators(df)
    df = df.dropna()
    
    # Step 3: Prepare features
    logger.info("Step 3: Preparing features...")
    feature_cols = ['log_return', 'RSI', 'MACD', 'SMA_ratio', 'volatility']
    data = df[feature_cols].values
    
    # Step 4: Walk-forward training
    logger.info("Step 4: Training model with walk-forward validation...")
    validator = WalkForwardValidator(
        min_train_window=config['walk_forward']['min_train_window'],
        step_size=config['walk_forward']['step_size'],
        n_splits=config['walk_forward']['n_splits']
    )
    
    all_predictions = []
    all_actuals = []
    
    for fold, (train_idx, test_idx) in enumerate(validator.split(data)):
        logger.info(f"Fold {fold + 1}...")
        # ... training code ...
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
    
    # Step 5: Backtest
    logger.info("Step 5: Backtesting strategy...")
    backtester = Backtester(
        initial_capital=config['backtest']['initial_capital'],
        transaction_cost_bps=config['backtest']['transaction_cost_bps']
    )
    
    results = backtester.run(
        predictions=all_predictions,
        actual_returns=all_actuals
    )
    
    # Step 6: Report
    logger.info("Step 6: Generating report...")
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Oracle Trading Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    main(args.ticker, args.start, args.end, config)
```

## 3.2 Using Click (Alternative to argparse)

```python
import click

@click.command()
@click.option('--ticker', default='AAPL', help='Stock ticker symbol')
@click.option('--start', default='2018-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end', default='2024-01-01', help='End date (YYYY-MM-DD)')
@click.option('--config', default='config/config.yaml', help='Path to config file')
def main(ticker, start, end, config):
    """Market Oracle - ML Trading Pipeline"""
    # ... same logic ...

if __name__ == "__main__":
    main()
```

---

# Part 4: CI/CD Pipeline (Day 26)

## 4.1 GitHub Actions Basics

```yaml
# .github/workflows/ci.yml

name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Type checking
      run: |
        mypy *.py --ignore-missing-imports
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: coverage.xml
```

## 4.2 CI/CD Workflow Visualization

```
PUSH TO GITHUB ──► GITHUB ACTIONS TRIGGERED
                           │
                           ▼
                   ┌───────────────┐
                   │ Checkout Code │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ Setup Python  │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ Install Deps  │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  Run Tests    │──► FAIL? ──► ❌ Red Badge
                   └───────┬───────┘
                           │ PASS
                           ▼
                   ┌───────────────┐
                   │  Type Check   │──► FAIL? ──► ❌ Red Badge
                   └───────┬───────┘
                           │ PASS
                           ▼
                   ┌───────────────┐
                   │   Coverage    │
                   └───────┬───────┘
                           │
                           ▼
                    ✅ Green Badge!
```

## 4.3 Adding Badges to README

```markdown
# Market Oracle 🔮

[![CI](https://github.com/YOUR_USERNAME/Market-Oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/Market-Oracle/actions)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/Market-Oracle/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/Market-Oracle)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"What is CI/CD and why is it important?"**
   - CI: Continuous Integration - auto-test on every push
   - CD: Continuous Deployment - auto-deploy after tests pass
   - Catches bugs early, ensures code quality

2. **"How do you handle secrets in CI/CD?"**
   - GitHub Secrets (encrypted environment variables)
   - Never commit API keys to repo
   - Reference secrets: `${{ secrets.API_KEY }}`

3. **"What's the difference between argparse and click?"**
   - argparse: Built-in, more verbose
   - click: Third-party, decorator-based, cleaner
   - Both work fine, click is more Pythonic

4. **"How do you structure a Python CLI application?"**
   - main() function with clear steps
   - Parse args at top
   - Load config from file
   - Have sensible defaults

## ML-Focused Questions

1. **"What's the Sharpe ratio and why is it important?"**
   - Return per unit of risk
   - Allows comparing strategies with different volatilities
   - Industry standard metric

2. **"Why include transaction costs in backtesting?"**
   - Real trading has costs (commissions, spreads)
   - High-frequency strategies may look great without costs
   - Realistic backtests require realistic costs

3. **"What's maximum drawdown and why does it matter?"**
   - Worst peak-to-trough decline
   - Measures risk of the strategy
   - Investors care about how bad it can get

4. **"How would you compare your strategy to a benchmark?"**
   - Compare Sharpe ratios (not just returns)
   - Same time period, same data
   - Account for transaction costs in both

5. **"What's a common backtesting mistake?"**
   - Lookahead bias (using future data)
   - Survivorship bias (only testing on stocks that exist today)
   - Ignoring transaction costs
   - Overfitting to historical data

---

# ✅ Day 23-26 Checklist

## Day 23 Deliverables
- [ ] Watch backtesting tutorial
- [ ] Create `backtester.py`
- [ ] Implement `generate_signals()`
- [ ] Implement `apply_transaction_costs()`
- [ ] Implement `compute_equity_curve()`
- [ ] Create `tests/test_backtester.py`

## Day 24 Deliverables
- [ ] Watch Sharpe ratio video
- [ ] Implement `compute_cagr()`
- [ ] Implement `compute_sharpe()`
- [ ] Implement `compute_max_drawdown()`
- [ ] Implement `compute_all_metrics()`
- [ ] Compare strategy vs buy-and-hold

## Day 25 Deliverables
- [ ] Watch argparse tutorial
- [ ] Create `main.py`
- [ ] Implement CLI with argparse/click
- [ ] Integrate all pipeline steps
- [ ] Test full pipeline runs end-to-end

## Day 26 Deliverables
- [ ] Watch GitHub Actions video
- [ ] Create `.github/workflows/ci.yml`
- [ ] Add pytest step
- [ ] Add mypy step
- [ ] Add coverage reporting
- [ ] Verify badge appears in README
- [ ] Push and verify workflow runs green

---

# 📖 Quick Reference

```python
# Backtesting
signals = (predictions > 0).astype(int)
strategy_returns = actual_returns * signals
equity = initial_capital * (1 + strategy_returns).cumprod()

# Performance Metrics
cagr = (final/initial) ** (1/years) - 1
sharpe = (return - rf) / volatility
drawdown = (peak - current) / peak

# CLI with argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="AAPL")
args = parser.parse_args()

# GitHub Actions trigger
on:
  push:
    branches: [main]
```

---

Onto the final stretch - Docker and polish! 🚀
