# 📚 Phase 1 Notes (Part 1): Days 1-3
## Data Infrastructure + Technical Indicators

> **Duration:** ~8-10 hours total  
> **Goal:** Build data pipeline and compute technical indicators  
> **Output:** `data_loader.py`, `indicators.py` (fully working with tests)

---

# 🗺️ Learning Roadmap

```
Day 1-2: Data Loading & Cleaning
├── yfinance API
├── pandas fundamentals
├── Handling missing data
└── Log returns

Day 3: Technical Indicators
├── RSI (momentum)
├── MACD (trend)
├── SMA (trend)
└── Volatility (risk)
```

---

# 📺 Videos to Watch FIRST

| Order | Topic | Video | Duration |
|-------|-------|-------|----------|
| 1 | pandas basics | [Corey Schafer - pandas Part 1](https://youtu.be/ZyhVh-qRZPA) | 35 min |
| 2 | pandas indexing | [Corey Schafer - pandas Part 2](https://youtu.be/W9XjRYFkkyw) | 30 min |
| 3 | pandas filtering | [Corey Schafer - pandas Part 3](https://youtu.be/Lw2rlcxScZY) | 25 min |
| 4 | RSI explained | [RSI Indicator Explained](https://youtu.be/oLXTl_Sw2O0) | 8 min |
| 5 | MACD explained | [MACD Indicator Explained](https://youtu.be/eob4wv2v--k) | 10 min |

**Optional (Recommended):**
- [pytest in 5 minutes](https://youtu.be/etosV2IWBF0) - For writing tests

---

# Part 1: Data Loading (Days 1-2)

## 1.1 Understanding OHLCV Data

Every trading day produces 5 key values:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRADING DAY                              │
│                                                              │
│   High ─────────────────●───────────────── (Peak price)     │
│                        /│\                                   │
│                       / │ \                                  │
│   Open ──────●───────/  │  \                                │
│              │      /   │   \                                │
│              │     /    │    \                               │
│              │    /     │     \                              │
│              │   /      │      \────────●── Close            │
│              │  /       │                                    │
│   Low ───────┴─/────────●────────────────── (Lowest price)  │
│                                                              │
│   Volume = Total shares traded (shown as bars usually)       │
└─────────────────────────────────────────────────────────────┘
```

| Column | What It Means | Why It Matters |
|--------|---------------|----------------|
| **Open** | First trade of the day | Shows overnight sentiment |
| **High** | Maximum price | Resistance level |
| **Low** | Minimum price | Support level |
| **Close** | Last trade of the day | Most important for analysis |
| **Volume** | Shares traded | Confirms price movements |

> [!IMPORTANT]
> **Adjusted Close** accounts for stock splits and dividends. For ML, prefer Adjusted Close or be consistent with regular Close.

---

## 1.2 yfinance Deep Dive

### Basic Usage Pattern

```python
import yfinance as yf

# Download single ticker
df = yf.download(
    tickers="AAPL",
    start="2020-01-01",
    end="2024-01-01",
    interval="1d"  # daily data
)

# Download multiple tickers
df = yf.download(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start="2020-01-01",
    end="2024-01-01"
)
```

### What yfinance Returns

```python
print(df.columns)
# Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')

print(df.index)
# DatetimeIndex(['2020-01-02', '2020-01-03', ...], dtype='datetime64[ns]')
```

### Key Parameters

| Parameter | Options | Use Case |
|-----------|---------|----------|
| `interval` | "1d", "1h", "5m", "1wk" | Granularity |
| `period` | "1y", "5y", "max" | Alternative to start/end |
| `auto_adjust` | True/False | Auto-adjust for splits |
| `progress` | True/False | Show download progress |

---

## 1.3 pandas Essentials

### Series vs DataFrame

```
┌──────────────────────────────────────────────────────────────┐
│                        DataFrame                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         Open    High    Low     Close   Volume          │ │
│  │ Date                                                     │ │
│  │ 2024-01-02  185.10  186.50  184.80  186.20  50000000   │ │
│  │ 2024-01-03  186.00  187.20  185.50  185.80  45000000   │ │
│  │ 2024-01-04  185.50  186.80  184.20  186.50  52000000   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                     │                                         │
│                     ▼                                         │
│              df["Close"]                                      │
│    ┌──────────────────────┐                                  │
│    │ Series (1D)          │                                  │
│    │ 2024-01-02  186.20   │                                  │
│    │ 2024-01-03  185.80   │                                  │
│    │ 2024-01-04  186.50   │                                  │
│    └──────────────────────┘                                  │
└──────────────────────────────────────────────────────────────┘
```

### Critical Operations

#### 1. The `.shift()` Method (You'll use this A LOT)

```python
# Original data
prices = pd.Series([100, 105, 103, 108])

# Shift forward (look at yesterday)
prices.shift(1)  # [NaN, 100, 105, 103]

# Shift backward (look at tomorrow)  
prices.shift(-1)  # [105, 103, 108, NaN]
```

Visual:
```
Index:     0     1     2     3
Original: 100   105   103   108
shift(1): NaN   100   105   103   ← Each value is "yesterday's"
shift(-1): 105  103   108   NaN   ← Each value is "tomorrow's"
```

#### 2. Handling Missing Data

```python
# Check for NaN
df.isna().sum()           # Count NaN per column
df.isnull().any()         # Any NaN in column?

# Strategy 1: Forward fill (best for time series)
df.fillna(method='ffill')  # Use previous value

# Strategy 2: Drop rows
df.dropna()                # Remove rows with any NaN

# Strategy 3: Fill with value
df.fillna(0)               # Replace with 0
```

> [!WARNING]
> For stock prices, **never** use backward fill (`bfill`) - that's future data leakage!

#### 3. Rolling Windows

```python
# 20-day moving average
df["Close"].rolling(window=20).mean()

# 20-day standard deviation
df["Close"].rolling(window=20).std()
```

---

## 1.4 Log Returns Theory

### Why Log Returns > Simple Returns?

| Property | Simple Return | Log Return |
|----------|--------------|------------|
| Formula | (P₁ - P₀) / P₀ | ln(P₁ / P₀) |
| Additivity | ❌ Not additive | ✅ Additive over time |
| Symmetry | ❌ Asymmetric | ✅ Symmetric |
| Distribution | Bounded at -100% | Unbounded |

### The Additivity Property (Key Insight!)

```
Day 1: Price goes 100 → 110 (+10%)
Day 2: Price goes 110 → 100 (-9.09%)

Simple returns: +10% + (-9.09%) = +0.91% ❌ (but you're back to 100!)

Log returns: ln(110/100) + ln(100/110) = 0.0953 + (-0.0953) = 0 ✅
```

### Formula Implementation

```python
import numpy as np

# These are mathematically equivalent:
log_return = np.log(df["Close"] / df["Close"].shift(1))
log_return = np.log(df["Close"]) - np.log(df["Close"].shift(1))
```

### Interpretation

| Log Return | Approximate % Change |
|------------|---------------------|
| 0.01 | +1% |
| 0.05 | +5.1% |
| 0.10 | +10.5% |
| -0.01 | -1% |
| -0.05 | -4.9% |
| -0.10 | -9.5% |

> [!TIP]
> For small returns (< 10%), log return ≈ simple return

---

# Part 2: Technical Indicators (Day 3)

## 2.1 Why Technical Indicators?

Raw OHLCV → Features ML can learn from

```
┌─────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING                           │
│                                                              │
│   OHLCV Data ──┬──► RSI (momentum)                          │
│                ├──► MACD (trend strength)                    │
│                ├──► SMA (trend direction)                    │
│                └──► Volatility (risk)                        │
│                              │                               │
│                              ▼                               │
│                     Feature Matrix X                         │
│                     (input to ML model)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2.2 RSI (Relative Strength Index)

### Concept
RSI measures **momentum** - how fast price is moving up vs down.

### Formula (Step by Step)

```
Step 1: Calculate price changes
    Δ = Close(t) - Close(t-1)

Step 2: Separate gains and losses
    Gain = max(Δ, 0)
    Loss = max(-Δ, 0)  # Absolute value

Step 3: Calculate average gain/loss (14-day EMA typically)
    Avg Gain = EMA(Gain, 14)
    Avg Loss = EMA(Loss, 14)

Step 4: Calculate RS (Relative Strength)
    RS = Avg Gain / Avg Loss

Step 5: Convert to RSI (0-100 scale)
    RSI = 100 - (100 / (1 + RS))
```

### Visualization

```
RSI Scale:
100 ├── Extremely Overbought (rare)
 80 ├── Overbought zone starts
 70 ├── ═══════════════════════════  ← Common sell signal
    │
 50 ├── Neutral
    │
 30 ├── ═══════════════════════════  ← Common buy signal
 20 ├── Oversold zone starts
  0 ├── Extremely Oversold (rare)
```

### Trading Interpretation

| RSI Value | Interpretation |
|-----------|----------------|
| > 70 | Overbought - might fall soon |
| 50-70 | Bullish territory |
| 30-50 | Bearish territory |
| < 30 | Oversold - might rise soon |

### Python Implementation Hint

```python
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    # 1. Calculate price changes
    delta = prices.diff()
    
    # 2. Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 3. Calculate rolling averages (use EMA for smoothing)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # 4. Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

---

## 2.3 MACD (Moving Average Convergence Divergence)

### Concept
MACD shows **trend direction and strength** by comparing fast and slow moving averages.

### Formula

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(MACD Line, 9)
Histogram = MACD Line - Signal Line
```

### Visualization

```
         Price Chart
         ───────────────────────────────
                    ╱\
                   ╱  \    ╱\
              ╱\  ╱    \  ╱  \
         ────╱  ╲╱      ╲╱    ────
         
         MACD Chart (Below)
         ───────────────────────────────
         
        MACD Line (fast, blue)
           ╱\        ╱\
          ╱  \      ╱  \
    ─────╱────╲────╱────╲─────── Zero Line
                  ╱
        Signal Line (slow, red)
           ╱\    ╱\
          ╱  \  ╱  \
    ─────╱────╲╱────────────────
         
        │ ▓▓▓ │     │ ░░░ │  ← Histogram (difference)
        └─────┘     └─────┘
        Bullish     Bearish
```

### Trading Signals

| Signal | Meaning |
|--------|---------|
| MACD crosses above Signal | Bullish (buy signal) |
| MACD crosses below Signal | Bearish (sell signal) |
| Histogram growing positive | Bullish momentum increasing |
| Histogram shrinking | Momentum weakening |

### Python Implementation Hint

```python
def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
```

---

## 2.4 SMA (Simple Moving Average)

### Concept
SMA smooths out price data to show the **trend direction**.

### Formula

```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

### Key SMAs

| SMA | Time Frame | Use |
|-----|------------|-----|
| SMA(20) | Short-term | Quick trends |
| SMA(50) | Medium-term | Common crossover |
| SMA(200) | Long-term | Major trend |

### Golden/Death Cross

```
Golden Cross (Bullish):
    SMA(50) crosses ABOVE SMA(200)
    
Death Cross (Bearish):
    SMA(50) crosses BELOW SMA(200)
```

```
         SMA(50)
           ╲
            ╲
             ╲
              ╳ ← Golden Cross (buy signal)
             ╱
            ╱
           ╱
         SMA(200)
```

### Python Implementation Hint

```python
def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()
```

---

## 2.5 Rolling Volatility

### Concept
Volatility measures **risk** - how much the price fluctuates.

### Formula

```
Volatility = std(log_returns, window=20)
```

### Interpretation

| Volatility | Meaning |
|------------|---------|
| Low (< 1%) | Calm market |
| Medium (1-2%) | Normal conditions |
| High (> 2%) | High uncertainty/risk |

### Python Implementation Hint

```python
def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window=window).std()
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"How would you handle missing data in a time series?"**
   - Forward fill for prices (no future leakage)
   - Drop first few rows after computing indicators (they'll have NaN)
   - Never backward fill!

2. **"What's the time complexity of computing a rolling mean?"**
   - O(n) if using a sliding window sum
   - pandas does this efficiently internally

3. **"How would you test a data loading function?"**
   - Test valid ticker returns DataFrame
   - Test invalid ticker handling
   - Test date range filtering
   - Test column presence
   - Mock API calls for unit tests

4. **"What's the difference between `.loc` and `.iloc`?"**
   - `.loc` uses labels (dates, column names)
   - `.iloc` uses integer positions

## ML-Focused Questions

1. **"Why use log returns instead of simple returns?"**
   - Additive over time
   - More normally distributed
   - Symmetric gains/losses
   - Better for statistical modeling

2. **"What is RSI and when would it fail?"**
   - Measures momentum (0-100)
   - Fails in strong trends (stays overbought/oversold)
   - Works best in ranging markets

3. **"Why are SMA crossovers considered lagging indicators?"**
   - They use historical data only
   - Signal comes AFTER the price has already moved
   - Good for confirmation, not prediction

4. **"What's the difference between EMA and SMA?"**
   - SMA: Equal weight to all periods
   - EMA: More weight to recent data
   - EMA reacts faster to price changes

---

# ✅ Day 1-3 Checklist

## Day 1-2 Deliverables
- [ ] Watch pandas videos (Parts 1-3)
- [ ] Implement `download_ticker_data()` in `data_loader.py`
- [ ] Implement `clean_data()` in `data_loader.py`
- [ ] Implement `compute_log_returns()` in `data_loader.py`
- [ ] Write tests in `test_data_loader.py`
- [ ] Run `pytest tests/test_data_loader.py -v`

## Day 3 Deliverables
- [ ] Watch RSI and MACD explanation videos
- [ ] Implement `compute_rsi()` in `indicators.py`
- [ ] Implement `compute_macd()` in `indicators.py`
- [ ] Implement `compute_sma()` in `indicators.py`
- [ ] Implement `compute_rolling_volatility()` in `indicators.py`
- [ ] Implement `add_all_indicators()` in `indicators.py`
- [ ] Write tests in `test_indicators.py`

---

# 📖 Quick Reference

```python
# yfinance
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Log returns
log_ret = np.log(df["Close"] / df["Close"].shift(1))

# RSI
delta = prices.diff()
gain = delta.where(delta > 0, 0)
loss = (-delta).where(delta < 0, 0)
rs = gain.ewm(span=14).mean() / loss.ewm(span=14).mean()
rsi = 100 - (100 / (1 + rs))

# MACD
macd = prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()

# SMA
sma = prices.rolling(window=50).mean()

# Volatility
vol = returns.rolling(window=20).std()
```

---

Now go implement! Ask me when you get stuck. 🚀
