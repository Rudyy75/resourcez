# 📚 Phase 2 Notes (Part 2): Days 12-14
## LSTM Training Pipeline + Evaluation

> **Duration:** ~8-10 hours total  
> **Goal:** Train LSTM with walk-forward, evaluate metrics, ensure code quality  
> **Output:** Training notebook, learning curves, model comparison

---

# 🗺️ Learning Roadmap

```
Day 12: Training Pipeline
├── Setting up Colab notebook
├── Integrating walk-forward with LSTM
├── Training across multiple folds

Day 13: Evaluation & Visualization
├── RMSE, MAE metrics
├── Direction accuracy
├── Learning curves
├── Predicted vs Actual plots

Day 14: Code Quality
├── Type hints
├── mypy type checking
├── Documentation
└── README update
```

---

# 📺 Videos to Watch

| Order | Topic | Video | Duration |
|-------|-------|-------|----------|
| 1 | TensorFlow Time Series | [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series) | 30 min |
| 2 | Learning Curves | [Learning Curves Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) | 15 min read |
| 3 | Type Hints | [ArjanCodes - Type Hints](https://youtu.be/QORvB-_mbZ0) | 15 min |
| 4 | mypy Tutorial | [mypy Tutorial](https://youtu.be/lle1x1kqIu0) | 18 min |

---

# Part 1: Training on Colab (Day 12)

## 1.1 Colab Notebook Setup

```python
# Cell 1: Mount Drive and Clone
from google.colab import drive
drive.mount('/content/drive')

!rm -rf Market-Oracle  # Fresh clone
!git clone https://github.com/YOUR_USERNAME/Market-Oracle.git
%cd Market-Oracle

# Cell 2: Install dependencies
!pip install -r requirements-colab.txt -q

# Cell 3: Import your modules
import sys
sys.path.append('/content/Market-Oracle')

from data_loader import download_ticker_data, clean_data, compute_log_returns
from indicators import add_all_indicators
from windowing import create_windows
from walk_forward import WalkForwardValidator
```

## 1.2 Full Training Pipeline

```python
# Cell 4: Data Preparation
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
ticker = "AAPL"
df = download_ticker_data(ticker, "2015-01-01", "2024-01-01")
df = clean_data(df)
df["log_return"] = compute_log_returns(df)
df = add_all_indicators(df)
df = df.dropna()

# Define features
feature_cols = ['log_return', 'RSI', 'MACD', 'SMA_ratio', 'volatility']
data = df[feature_cols].values

print(f"Data shape: {data.shape}")
print(f"Features: {feature_cols}")
```

```python
# Cell 5: Walk-Forward Training Loop
from tensorflow.keras.callbacks import EarlyStopping
from models.lstm_model import build_lstm_model

WINDOW_SIZE = 30
results = []

validator = WalkForwardValidator(
    min_train_window=252,
    step_size=63,  # ~3 months
    n_splits=8
)

for fold, (train_idx, test_idx) in enumerate(validator.split(data)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}")
    print(f"{'='*50}")
    print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    
    # Split data
    train_data = data[train_idx]
    test_data = data[test_idx]
    
    # Scale features (fit on train only!)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create windows
    X_train, y_train = create_windows(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_windows(test_scaled, WINDOW_SIZE)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Build fresh model for each fold
    model = build_lstm_model(input_shape=(WINDOW_SIZE, len(feature_cols)))
    
    # Train
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Predict
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Direction accuracy
    dir_actual = (y_test > 0).astype(int)
    dir_pred = (y_pred > 0).astype(int)
    dir_acc = (dir_actual == dir_pred).mean()
    
    results.append({
        'fold': fold + 1,
        'rmse': rmse,
        'mae': mae,
        'dir_accuracy': dir_acc,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx)
    })
    
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {dir_acc:.2%}")

# Summary
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("OVERALL RESULTS")
print("="*50)
print(results_df.to_string(index=False))
print(f"\nMean RMSE: {results_df['rmse'].mean():.6f}")
print(f"Mean Direction Accuracy: {results_df['dir_accuracy'].mean():.2%}")
```

---

# Part 2: Evaluation Metrics (Day 13)

## 2.1 Regression Metrics

### RMSE (Root Mean Square Error)

```
RMSE = √(Σ(yᵢ - ŷᵢ)² / n)

Interpretation:
- Units same as target (log returns)
- Penalizes large errors more (squared)
- RMSE = 0.02 means average error of ~2% in log return
```

### MAE (Mean Absolute Error)

```
MAE = Σ|yᵢ - ŷᵢ| / n

Interpretation:
- Units same as target
- Treats all errors equally
- More robust to outliers than RMSE
```

### When to Use Which?

| Metric | Use When |
|--------|----------|
| RMSE | Large errors are particularly bad |
| MAE | All errors equally important |
| Both | Report both for completeness |

## 2.2 Direction Accuracy

```python
def direction_accuracy(y_true, y_pred):
    """
    Percentage of times we predicted the correct direction.
    
    This is arguably MORE important than RMSE for trading!
    """
    actual_direction = (y_true > 0).astype(int)  # 1 = UP, 0 = DOWN
    pred_direction = (y_pred > 0).astype(int)
    
    return (actual_direction == pred_direction).mean()
```

```
Why Direction Accuracy Matters:

Scenario A: RMSE = 0.01, Direction Accuracy = 50%
  You predict magnitude well but randomly guess direction.
  Trading result: RANDOM!

Scenario B: RMSE = 0.03, Direction Accuracy = 55%
  You predict magnitude poorly but direction slightly better.
  Trading result: PROFITABLE! (55% > 50%)
  
For trading, direction > magnitude!
```

## 2.3 Learning Curves

```python
import matplotlib.pyplot as plt

def plot_learning_curves(history, fold):
    """Plot train vs validation loss."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Fold {fold}: Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title(f'Fold {fold}: MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'outputs/figures/learning_curve_fold{fold}.png')
    plt.show()
```

### Interpreting Learning Curves

```
GOOD (Well-fitted):                BAD (Overfitting):
                                   
Loss │                            Loss │
     │ ────                            │ ────  train
     │     ────____                    │     ────___________
     │             ~~~~  val           │              _____  ← val increasing!
     └──────────────►                  └──────────────►
       Epochs                            Epochs
     
Both curves converge → Good!       Gap widens → Overfitting!


BAD (Underfitting):                BAD (Not converged):
                                   
Loss │                            Loss │
     │ ____________  train            │ ─
     │ ____________  val              │  ─               ← still dropping
     │                                 │   ─────
     └──────────────►                  └──────────────►
       Epochs                            Epochs
     
Both high and flat → Underfitting!  Lost still dropping → Train longer!
```

## 2.4 Predicted vs Actual Plot

```python
def plot_predictions(y_true, y_pred, fold):
    """Scatter plot of predicted vs actual returns."""
    plt.figure(figsize=(8, 8))
    
    # Scatter
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    
    # Zero lines (quadrant boundaries)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.xlabel('Actual Log Return')
    plt.ylabel('Predicted Log Return')
    plt.title(f'Fold {fold}: Predicted vs Actual')
    plt.legend()
    
    # Add quadrant labels
    plt.text(0.02, 0.02, 'True Positives', fontsize=10, alpha=0.5)
    plt.text(-0.04, 0.02, 'False Positives', fontsize=10, alpha=0.5)
    plt.text(-0.04, -0.02, 'True Negatives', fontsize=10, alpha=0.5)
    plt.text(0.02, -0.02, 'False Negatives', fontsize=10, alpha=0.5)
    
    plt.savefig(f'outputs/figures/pred_vs_actual_fold{fold}.png')
    plt.show()
```

### Interpreting the Plot

```
                    Predicted Return
                         ▲
                         │
           FP(bad)       │      TP(good)
         Predicted UP    │    Predicted UP
         Actually DOWN   │    Actually UP
                         │
    ─────────────────────┼─────────────────► Actual Return
                         │
           TN(good)      │      FN(bad)
         Predicted DOWN  │    Predicted DOWN
         Actually DOWN   │    Actually UP
                         │
```

---

# Part 3: Code Quality (Day 14)

## 3.1 Type Hints

```python
# ❌ WITHOUT TYPE HINTS
def create_windows(data, window_size):
    # What's data? Array? List? DataFrame?
    pass

# ✅ WITH TYPE HINTS
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

def create_windows(
    data: NDArray[np.float64],
    window_size: int = 30
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert time series to windowed format.
    
    Args:
        data: Shape (n_samples, n_features)
        window_size: Number of timesteps per window
        
    Returns:
        X: Shape (n_windows, window_size, n_features)
        y: Shape (n_windows,)
    """
    pass
```

### Common Type Annotations

```python
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}"

# Optional (can be None)
def fetch_data(ticker: str, cache: Optional[bool] = None) -> pd.DataFrame:
    pass

# Collections
def get_tickers() -> List[str]:
    return ["AAPL", "MSFT"]

def get_config() -> Dict[str, int]:
    return {"window": 30, "epochs": 100}

# Tuple with specific types
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df[:100], df[100:]

# Union (multiple types)
def process(data: Union[list, np.ndarray]) -> np.ndarray:
    return np.array(data)
```

## 3.2 Running mypy

```bash
# Install mypy
pip install mypy

# Run on single file
mypy data_loader.py

# Run on all source files
mypy *.py --ignore-missing-imports

# Strict mode (more checks)
mypy *.py --strict
```

### Common mypy Errors

```
# Error: Missing return type
def fetch():  # Missing return type
    return 42
# Fix: def fetch() -> int:

# Error: Incompatible types
x: int = "hello"  # Can't assign str to int
# Fix: x: str = "hello" or x: int = 42

# Error: None vs Optional
def get(key: str) -> str:  # Should be Optional[str]
    return cache.get(key)  # Might return None!
# Fix: def get(key: str) -> Optional[str]:
```

## 3.3 Docstrings (Google Style)

```python
def compute_rsi(
    prices: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    RSI is a momentum indicator that measures the speed and change
    of price movements. Values range from 0 to 100.
    
    Args:
        prices: Series of closing prices indexed by date.
        period: Lookback period for RSI calculation. Default 14.
    
    Returns:
        Series of RSI values with same index as input.
        First `period` values will be NaN.
    
    Raises:
        ValueError: If prices is empty or period < 1.
    
    Example:
        >>> prices = pd.Series([100, 102, 101, 105])
        >>> rsi = compute_rsi(prices, period=3)
        >>> print(rsi.iloc[-1])
        65.4
    """
    pass
```

---

# Part 4: Model Comparison

## 4.1 Phase 1 vs Phase 2 Comparison Table

```python
# Create comparison table
comparison = pd.DataFrame({
    'Model': ['Naive (Always UP)', 'Logistic Regression', 'Random Forest', 'LSTM'],
    'Direction Accuracy': [0.52, 0.54, 0.55, 0.56],
    'RMSE': ['-', 0.021, 0.020, 0.018],
    'Lift vs Baseline': ['-', '+2%', '+3%', '+4%']
})

print(comparison.to_markdown(index=False))
```

| Model | Direction Accuracy | RMSE | Lift vs Baseline |
|-------|-------------------|------|------------------|
| Naive (Always UP) | 52% | - | - |
| Logistic Regression | 54% | 0.021 | +2% |
| Random Forest | 55% | 0.020 | +3% |
| LSTM | 56% | 0.018 | +4% |

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"What's the benefit of type hints in Python?"**
   - Catch bugs before runtime (with mypy)
   - Self-documenting code
   - Better IDE autocomplete
   - Helps with team collaboration

2. **"How would you structure a training notebook for production?"**
   - Clear sections with markdown headers
   - Config at top (hyperparameters)
   - Reproducible (set random seeds)
   - Save results and models
   - Include visualizations

3. **"What are callbacks in Keras?"**
   - Functions called during training
   - EarlyStopping: stop when no improvement
   - ModelCheckpoint: save best model
   - TensorBoard: visualization

4. **"How do you ensure reproducibility in ML experiments?"**
   ```python
   import numpy as np
   import tensorflow as tf
   import random
   
   SEED = 42
   np.random.seed(SEED)
   tf.random.set_seed(SEED)
   random.seed(SEED)
   ```

## ML-Focused Questions

1. **"What's the difference between RMSE and MAE?"**
   - RMSE penalizes large errors more (squared)
   - MAE treats all errors equally
   - RMSE ≥ MAE always
   - Use RMSE when large errors are worse

2. **"Why might direction accuracy be more important than RMSE for trading?"**
   - Trading profits depend on direction, not magnitude
   - 55% direction accuracy + proper sizing = profitable
   - Perfect RMSE with 50% direction = random trading

3. **"What do learning curves tell you?"**
   - Overfitting: train drops, val increases
   - Underfitting: both stay high
   - Good fit: both converge

4. **"How do you choose when to stop training?"**
   - Early stopping based on validation loss
   - Patience: how many epochs to wait
   - Restore best weights, not last weights

5. **"Why train a new model for each walk-forward fold?"**
   - Each fold has different train/test split
   - Prevents data leakage between folds
   - More realistic simulation of real-world retraining

---

# ✅ Day 12-14 Checklist

## Day 12 Deliverables
- [ ] Set up Colab notebook (`02_lstm_training.ipynb`)
- [ ] Implement full training pipeline
- [ ] Integrate walk-forward with LSTM
- [ ] Train across 8+ folds
- [ ] Log results per fold

## Day 13 Deliverables
- [ ] Calculate RMSE, MAE for each fold
- [ ] Calculate direction accuracy
- [ ] Plot learning curves per fold
- [ ] Plot predicted vs actual scatter
- [ ] Create model comparison table
- [ ] Save figures to `outputs/figures/`

## Day 14 Deliverables
- [ ] Add type hints to all functions
- [ ] Run `mypy *.py --ignore-missing-imports`
- [ ] Fix any type errors
- [ ] Write docstrings for all public functions
- [ ] Update README with Phase 2 results
- [ ] Push to GitHub: `git push origin main`

---

# 📖 Quick Reference

```python
# Walk-forward with LSTM
for fold, (train_idx, test_idx) in enumerate(validator.split(data)):
    # Fresh scaler per fold
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create windows
    X_train, y_train = create_windows(train_scaled, 30)
    X_test, y_test = create_windows(test_scaled, 30)
    
    # Fresh model per fold
    model = build_lstm_model(input_shape=(30, 5))
    model.fit(X_train, y_train, callbacks=[early_stop])

# Metrics
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
dir_acc = ((y_true > 0) == (y_pred > 0)).mean()

# Type hints
from typing import Tuple, Optional
def func(x: int, y: Optional[str] = None) -> Tuple[int, str]:
    pass

# mypy
mypy *.py --ignore-missing-imports
```

---

Phase 2 complete! You now have a trained LSTM. Next: Attention mechanism! 🚀
