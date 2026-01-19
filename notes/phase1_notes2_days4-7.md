# 📚 Phase 1 Notes (Part 2): Days 4-7
## Testing, Walk-Forward Validation, and Classifiers

> **Duration:** ~8-10 hours total  
> **Goal:** Test your code, implement walk-forward validation, train baseline classifiers  
> **Output:** Tests passing, `walk_forward.py`, notebook with LR/RF models

---

# 🗺️ Learning Roadmap

```
Day 4: Testing Indicators
├── Writing pytest tests
├── Testing edge cases
└── Validating against external sources

Day 5: Walk-Forward Validation
├── Why random splits fail
├── Expanding window approach
└── Time-series cross-validation

Day 6: Phase 1 Models
├── Logistic Regression
├── Random Forest
├── Feature importance
└── Naive baseline

Day 7: Documentation & Review
├── Docstrings
├── README updates
└── Git workflow
```

---

# 📺 Videos to Watch FIRST

| Order | Topic | Video | Duration |
|-------|-------|-------|----------|
| 1 | pytest basics | [pytest Tutorial](https://youtu.be/YbpKMIUjvK8) | 45 min |
| 2 | Walk-forward | [Time Series CV](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/) | 15 min read |
| 3 | Logistic Regression | [StatQuest - Logistic Regression](https://youtu.be/yIYKR4sgzI8) | 9 min |
| 4 | Random Forest | [StatQuest - Random Forest](https://youtu.be/J4Wdy0Wc_xQ) | 10 min |

**Optional (Highly Recommended):**
- [StatQuest - Decision Trees](https://youtu.be/_L39rN6gz7Y) - Before Random Forest
- [StatQuest - Cross Validation](https://youtu.be/fSytzGwwBVw) - General CV concepts

---

# Part 1: Testing with pytest (Day 4)

## 1.1 Why Test?

```
┌─────────────────────────────────────────────────────────────┐
│                     WITHOUT TESTS                            │
│                                                              │
│   Write Code ──► "Looks good" ──► Deploy ──► 💥 BUG!        │
│                                                              │
│                     WITH TESTS                               │
│                                                              │
│   Write Code ──► Run Tests ──► Fix ──► Tests Pass ──► Ship  │
│                       │           ▲                          │
│                       └───────────┘                          │
│                       (iterate until green)                  │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 pytest Fundamentals

### Basic Test Structure

```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from indicators import compute_rsi, compute_sma

class TestRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_bounds(self):
        """RSI should always be between 0 and 100."""
        prices = pd.Series([100, 105, 102, 108, 106, 110, 107])
        rsi = compute_rsi(prices, period=3)
        
        # Drop NaN values from the beginning
        rsi_valid = rsi.dropna()
        
        assert rsi_valid.min() >= 0, "RSI should not be negative"
        assert rsi_valid.max() <= 100, "RSI should not exceed 100"
    
    def test_rsi_overbought(self):
        """Consistently rising prices should give high RSI."""
        # Price goes up every day
        prices = pd.Series([100, 105, 110, 115, 120, 125, 130])
        rsi = compute_rsi(prices, period=5)
        
        assert rsi.iloc[-1] > 70, "Rising prices should be overbought"
```

### Key pytest Concepts

| Concept | Usage |
|---------|-------|
| `assert` | Check if condition is True |
| `pytest.raises` | Expect an exception |
| `pytest.approx` | Compare floats with tolerance |
| `@pytest.fixture` | Reusable test setup |
| `@pytest.mark.parametrize` | Run same test with different inputs |

### Useful Assertions

```python
# Basic assertions
assert result == expected
assert len(df) > 0
assert "column_name" in df.columns

# Float comparison (with tolerance)
assert result == pytest.approx(expected, rel=1e-3)

# Exception handling
with pytest.raises(ValueError):
    compute_rsi(pd.Series([]))  # Empty series should raise error

# Check pandas objects
pd.testing.assert_series_equal(result, expected)
pd.testing.assert_frame_equal(result, expected)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_indicators.py -v

# Run specific test
pytest tests/test_indicators.py::TestRSI::test_rsi_bounds -v

# Show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x
```

## 1.3 What to Test for Indicators

```
┌──────────────────────────────────────────────────────────────┐
│                    TEST CATEGORIES                            │
│                                                               │
│  1. BOUNDS CHECKING                                           │
│     ├── RSI: 0 ≤ value ≤ 100                                 │
│     ├── Volatility: value ≥ 0                                │
│     └── SMA: Between min and max price                       │
│                                                               │
│  2. EDGE CASES                                                │
│     ├── Empty input                                          │
│     ├── Single value                                         │
│     ├── Insufficient data for period                         │
│     └── All same values (no change)                          │
│                                                               │
│  3. KNOWN VALUES                                              │
│     ├── SMA: Easy to verify manually                         │
│     ├── Compare with TradingView/external source             │
│     └── Use example from textbook                            │
│                                                               │
│  4. OUTPUT SHAPE                                              │
│     ├── Same length as input                                 │
│     ├── Correct number of NaN at start                       │
│     └── Correct dtypes                                       │
└──────────────────────────────────────────────────────────────┘
```

### Example: Testing SMA Manually

```python
def test_sma_manual_calculation(self):
    """Verify SMA against manual calculation."""
    prices = pd.Series([10, 20, 30, 40, 50])
    sma = compute_sma(prices, period=3)
    
    # SMA(3) at index 2 = (10 + 20 + 30) / 3 = 20
    assert sma.iloc[2] == pytest.approx(20.0)
    
    # SMA(3) at index 3 = (20 + 30 + 40) / 3 = 30
    assert sma.iloc[3] == pytest.approx(30.0)
    
    # First two values should be NaN (not enough data)
    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])
```

---

# Part 2: Walk-Forward Validation (Day 5)

## 2.1 Why Regular Cross-Validation Fails

```
❌ REGULAR K-FOLD (WRONG for Time Series!)
   
   Data: [Jan][Feb][Mar][Apr][May][Jun][Jul][Aug][Sep][Oct]
   
   Fold 1: Train on [Feb,Apr,Jun,Aug,Oct], Test on [Jan,Mar,May,Jul,Sep]
                                                      │
                                           Using FUTURE data to predict PAST!
                                           This is DATA LEAKAGE!

✅ WALK-FORWARD (CORRECT for Time Series!)

   Fold 1: Train [Jan─Feb─Mar], Test [Apr]
   Fold 2: Train [Jan─Feb─Mar─Apr], Test [May]  
   Fold 3: Train [Jan─Feb─Mar─Apr─May], Test [Jun]
   ...
   
   Always train on PAST, test on FUTURE!
```

## 2.2 Walk-Forward Types

### Expanding Window (What we'll use)

```
Training window GROWS each fold:

Fold 1: [████████░░░░░░░░░░░░]  Train=200, Test=20
Fold 2: [█████████████░░░░░░░]  Train=220, Test=20
Fold 3: [██████████████████░░]  Train=240, Test=20
        ─────────────────────►
                Time
```

### Sliding Window (Alternative)

```
Training window STAYS SAME SIZE:

Fold 1: [████████░░░░░░░░░░░░]  Train=200, Test=20
Fold 2: [   █████████░░░░░░░░]  Train=200, Test=20
Fold 3: [      ██████████░░░░]  Train=200, Test=20
        ─────────────────────►
                Time
```

## 2.3 Implementation Logic

```python
class WalkForwardValidator:
    def __init__(self, min_train_window=252, step_size=21, n_splits=10):
        """
        Args:
            min_train_window: Minimum training samples (252 = 1 year)
            step_size: How many samples between folds (21 = ~1 month)
            n_splits: Number of validation folds
        """
        self.min_train_window = min_train_window
        self.step_size = step_size
        self.n_splits = n_splits
    
    def split(self, X):
        n_samples = len(X)
        
        for i in range(self.n_splits):
            # Training end grows with each fold
            train_end = self.min_train_window + (i * self.step_size)
            test_end = train_end + self.step_size
            
            # Don't exceed data length
            if test_end > n_samples:
                break
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(train_end, test_end)
            
            yield train_indices, test_indices
```

### Visual Example

```
Total data: 500 samples
min_train_window: 252
step_size: 21

Fold 0: train[0:252],   test[252:273]  (21 samples)
Fold 1: train[0:273],   test[273:294]
Fold 2: train[0:294],   test[294:315]
...
Fold 9: train[0:441],   test[441:462]
```

## 2.4 Testing Walk-Forward

```python
def test_no_data_leakage(self):
    """Test indices never overlap."""
    X = np.arange(500)
    validator = WalkForwardValidator(min_train_window=100, step_size=20, n_splits=5)
    
    for train_idx, test_idx in validator.split(X):
        # Test indices should always be AFTER train indices
        assert test_idx.min() > train_idx.max(), "Data leakage detected!"
```

---

# Part 3: Classification Models (Day 6)

## 3.1 Problem Setup

We're predicting **direction** (binary classification):

```
Target (y):
  if log_return(t+1) > 0:
      y = 1  (UP)
  else:
      y = 0  (DOWN)

Features (X):
  - RSI
  - MACD
  - SMA_50 / SMA_200 ratio
  - Volatility
  - Previous log returns
```

## 3.2 Logistic Regression

### Concept

```
                    ┌─────────────────┐
  Features (X) ───► │  z = w·X + b    │ ───► σ(z) ───► P(UP)
                    └─────────────────┘
                           │
                           ▼
                    Sigmoid Function
                    
         1 │      ___________
           │     /
      0.5 │----/-------------- threshold
           │   /
         0 │__/
           └──────────────────►
              z (linear output)
```

### Formula

```
P(y=1|X) = σ(w·X + b) = 1 / (1 + e^(-(w·X + b)))

Predict 1 if P > 0.5, else predict 0
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Simple, interpretable | Assumes linear decision boundary |
| Fast training | May underfit complex patterns |
| Probabilities are calibrated | Needs feature scaling |

### scikit-learn Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale features (important for Logistic Regression!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
```

## 3.3 Random Forest

### Concept

```
                    ┌──────────────┐
                    │   Tree 1     │ ──► Vote: UP
                    └──────────────┘
                    
  Features ────────►┌──────────────┐
                    │   Tree 2     │ ──► Vote: DOWN  ──► Majority Vote ──► UP
                    └──────────────┘
                    
                    ┌──────────────┐
                    │   Tree 3     │ ──► Vote: UP
                    └──────────────┘
                    
                   (100+ trees typically)
```

### How Each Tree Works

```
                    Is RSI > 70?
                   /            \
                 Yes             No
                 /                \
        Is Vol > 2%?         Is MACD > 0?
        /         \          /          \
      Yes         No       Yes          No
       │           │         │            │
    DOWN(0.7)   UP(0.6)   UP(0.65)   DOWN(0.55)
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Handles non-linear relationships | Can overfit |
| No feature scaling needed | Slower training |
| Feature importance built-in | Less interpretable |
| Robust to outliers | |

### scikit-learn Usage

```python
from sklearn.ensemble import RandomForestClassifier

# Train (no scaling needed!)
model = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=10,       # Prevent overfitting
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)
```

## 3.4 Naive Baseline (Critical!)

### Why You NEED a Baseline

```
Your Model: 55% accuracy
Sounds good right?

But wait...
Naive Baseline (always predict UP): 54% accuracy

Your model is barely better than guessing! 📉
```

### Common Baselines

| Baseline | Strategy | When to use |
|----------|----------|-------------|
| Always Up | Predict 1 always | Bull markets |
| Always Down | Predict 0 always | Bear markets |
| Random | 50-50 guess | Reference |
| Previous Direction | Predict same as yesterday | Momentum check |

### Implementation

```python
# Baseline 1: Always predict UP
y_baseline_up = np.ones_like(y_test)
baseline_acc = (y_baseline_up == y_test).mean()

# Baseline 2: Previous direction
y_baseline_prev = y_test.shift(1).fillna(1)

print(f"Your model: {model_accuracy:.2%}")
print(f"Always UP baseline: {baseline_acc:.2%}")
print(f"Lift: {model_accuracy - baseline_acc:.2%}")
```

## 3.5 Evaluation Metrics

### Confusion Matrix

```
                    Predicted
                  UP       DOWN
              ┌────────┬────────┐
         UP   │   TP   │   FN   │
Actual        ├────────┼────────┤
        DOWN  │   FP   │   TN   │
              └────────┴────────┘

TP = True Positive (Predicted UP, was UP)
TN = True Negative (Predicted DOWN, was DOWN)
FP = False Positive (Predicted UP, was DOWN - bad!)
FN = False Negative (Predicted DOWN, was UP - missed opportunity)
```

### Key Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
# 
#          0       0.52      0.48      0.50       100
#          1       0.54      0.58      0.56       110
# 
#   accuracy                           0.53       210
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/(Total) | Overall correctness |
| **Precision** | TP/(TP+FP) | When I predict UP, how often right? |
| **Recall** | TP/(TP+FN) | Of all UPs, how many did I catch? |
| **F1** | 2×(P×R)/(P+R) | Balance of precision & recall |

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"How do you structure tests for a data pipeline?"**
   - Unit tests for each function
   - Integration tests for end-to-end flow  
   - Mock external API calls
   - Use fixtures for common test data

2. **"What's the difference between unit and integration tests?"**
   - Unit: Test one function in isolation
   - Integration: Test multiple components together
   - E2E: Test complete user flow

3. **"How would you make tests deterministic?"**
   - Set `random_state` in all random operations
   - Use fixed test data (not live API)
   - Mock time-dependent functions

4. **"What is test coverage and what's a good target?"**
   - % of code executed by tests
   - 80%+ is good, 100% is ideal but diminishing returns
   - Focus on critical paths

## ML-Focused Questions

1. **"Why can't you use k-fold CV for time series?"**
   - Causes data leakage (training on future)
   - Time has inherent ordering
   - Walk-forward respects temporal structure

2. **"What's the difference between Logistic Regression and Random Forest?"**
   - LR: Linear decision boundary, needs scaling, interpretable
   - RF: Non-linear, handles interactions, feature importance

3. **"How do you handle class imbalance in stock direction prediction?"**
   - Class weights (`class_weight='balanced'`)
   - Oversampling minority class (SMOTE)
   - Adjust decision threshold
   - Use F1 instead of accuracy

4. **"What's feature importance and how do you compute it?"**
   - RF: Based on mean decrease in impurity
   - LR: Magnitude of coefficients (after scaling)
   - Permutation importance: Drop each feature, measure accuracy drop

5. **"Why might a model with 55% accuracy still be valuable for trading?"**
   - If wins are larger than losses (asymmetric payoff)
   - Risk-adjusted returns matter more
   - Edge over baseline is what counts

---

# ✅ Day 4-7 Checklist

## Day 4 Deliverables
- [ ] Watch pytest tutorial
- [ ] Create `tests/test_indicators.py`
- [ ] Test RSI bounds (0-100)
- [ ] Test SMA manual calculation
- [ ] Test MACD components
- [ ] Test edge cases (empty, single value)
- [ ] All tests passing ✓

## Day 5 Deliverables
- [ ] Read Walk-Forward article
- [ ] Implement `WalkForwardValidator.split()` 
- [ ] Create `tests/test_walk_forward.py`
- [ ] Test no data leakage
- [ ] Test fold sizes correct

## Day 6 Deliverables
- [ ] Watch StatQuest videos (LR, RF)
- [ ] Create `notebooks/01_phase1_classifier.ipynb`
- [ ] Prepare feature matrix X and target y
- [ ] Train Logistic Regression
- [ ] Train Random Forest
- [ ] Compute naive baseline (always UP)
- [ ] Print classification reports
- [ ] Plot feature importance
- [ ] Save best model

## Day 7 Deliverables
- [ ] Add docstrings to all functions
- [ ] Update README with Phase 1 results
- [ ] Run `pytest tests/ -v` (all green)
- [ ] `git add .` → `git commit -m "Phase 1 complete"`
- [ ] Push to GitHub

---

# 📖 Quick Reference

```python
# pytest
pytest tests/ -v                    # Run all tests
pytest tests/test_file.py -v        # Run specific file
assert value == expected            # Basic assertion
assert value == pytest.approx(exp)  # Float comparison

# Walk-forward split
for train_idx, test_idx in validator.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model.feature_importances_

# Metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

Now go implement! Day 4: Focus on tests. Day 5: Walk-forward. Day 6: Models. Day 7: Polish! 🚀
