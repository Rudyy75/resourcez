# 📚 Phase 2 Notes (Part 1): Days 8-11
## Deep Learning Foundations + LSTM Architecture

> **Duration:** ~10-12 hours total  
> **Goal:** Understand neural networks, build LSTM model architecture  
> **Output:** `windowing.py`, `models/lstm_model.py`

---

# 🗺️ Learning Roadmap

```
Day 8: Windowing Pipeline
├── Why windows for time series
├── create_windows() function
├── Shape: (samples, timesteps, features)

Day 9: Windowing Tests + Scaling
├── Test output shapes
├── Feature scaling (MinMax/Standard)
├── Preventing data leakage in scaling

Day 10: LSTM Architecture
├── Neural network fundamentals
├── Why RNNs fail (vanishing gradient)
├── LSTM gates and cell state

Day 11: TensorFlow/Keras Basics
├── Sequential API
├── Building LSTM model
├── Logging infrastructure
```

---

# 📺 Videos to Watch FIRST (CRITICAL!)

| Order | Topic | Video | Duration | Priority |
|-------|-------|-------|----------|----------|
| 1 | Neural Networks | [3Blue1Brown - Neural Networks](https://youtu.be/aircAruvnKk) | 19 min | 🔴 Must |
| 2 | Backpropagation | [3Blue1Brown - Backprop](https://youtu.be/Ilg3gGewQ5U) | 14 min | 🔴 Must |
| 3 | RNN Intro | [StatQuest - RNN](https://youtu.be/AsNTP8Kwu80) | 15 min | 🔴 Must |
| 4 | LSTM Deep Dive | [StatQuest - LSTM](https://youtu.be/YCzL96nL7j0) | 13 min | 🔴 Must |
| 5 | LSTM Blog | [Colah's LSTM Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 20 min | 🔴 Must |
| 6 | Keras Sequential | [TensorFlow LSTM Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series) | 30 min | 🟡 Good |

> [!IMPORTANT]
> Do NOT skip the 3Blue1Brown and StatQuest videos. They explain the intuition behind deep learning better than any textbook.

---

# Part 1: Windowing for Time Series (Days 8-9)

## 1.1 Why Windows?

Neural networks need fixed-size inputs. Time series are sequential.

```
RAW DATA (Sequential):
Day:    1    2    3    4    5    6    7    8    9   10
Price: 100  102  101  105  103  108  107  110  112  115

WINDOWED DATA (Fixed-size chunks):
Window 1: [100, 102, 101, 105, 103] → Predict: 108
Window 2: [102, 101, 105, 103, 108] → Predict: 107
Window 3: [101, 105, 103, 108, 107] → Predict: 110
Window 4: [105, 103, 108, 107, 110] → Predict: 112
...

Each window: 5 days of history → predict next day
```

## 1.2 The Shape Matters!

```
For LSTM, input shape is: (samples, timesteps, features)

            samples    timesteps   features
              │            │           │
              ▼            ▼           ▼
       ┌─────────────────────────────────────┐
       │                                     │
       │   [[[0.1, 0.5, 0.3, 0.2],          │  ◄── Sample 0
       │     [0.2, 0.4, 0.4, 0.1],          │      Window of 30 days
       │     ...                             │      4 features each day
       │     [0.15, 0.6, 0.35, 0.25]],      │
       │                                     │
       │    [[0.2, 0.4, 0.4, 0.1],          │  ◄── Sample 1
       │     [0.15, 0.5, 0.35, 0.2],        │
       │     ...                             │
       │     [0.18, 0.55, 0.38, 0.22]]]     │
       │                                     │
       └─────────────────────────────────────┘

X shape: (n_samples, 30, 4)
  - n_samples: number of windows we created
  - 30: window size (look back 30 days)
  - 4: number of features (RSI, MACD, etc.)

y shape: (n_samples,)
  - Target for each window (next day's return)
```

## 1.3 Implementation Logic

```python
def create_windows(data: np.ndarray, window_size: int = 30):
    """
    Convert time series to supervised learning format.
    
    Args:
        data: Array of shape (n_days, n_features)
        window_size: Number of days to look back
    
    Returns:
        X: shape (n_samples, window_size, n_features)
        y: shape (n_samples,)
    """
    X, y = [], []
    
    for i in range(window_size, len(data)):
        # Window: days [i-window_size, i-1]
        X.append(data[i-window_size:i])
        # Target: day i
        y.append(data[i, 0])  # Assuming target is first column
    
    return np.array(X), np.array(y)
```

### Visual Walkthrough

```
data = [[day0], [day1], [day2], [day3], [day4], [day5], [day6], ...]
window_size = 3

i=3: X.append([day0, day1, day2]), y.append(day3)
i=4: X.append([day1, day2, day3]), y.append(day4)
i=5: X.append([day2, day3, day4]), y.append(day5)
...

Result:
X[0] = [day0, day1, day2]  →  y[0] = day3
X[1] = [day1, day2, day3]  →  y[1] = day4
X[2] = [day2, day3, day4]  →  y[2] = day5
```

## 1.4 Feature Scaling (Critical for Neural Networks!)

### Why Scale?

```
UNSCALED FEATURES:
  RSI:        [30, 45, 70, 55]      (range: 0-100)
  MACD:       [0.5, -0.3, 1.2, 0.8] (range: -5 to 5)
  Volume:     [50M, 45M, 60M, 55M]  (range: millions!)
  
Problem: Volume dominates the gradients! Neural network
learns mostly from volume, ignores RSI and MACD.

SCALED FEATURES (all 0-1):
  RSI:        [0.30, 0.45, 0.70, 0.55]
  MACD:       [0.55, 0.47, 0.62, 0.58]
  Volume:     [0.33, 0.00, 1.00, 0.67]
  
Now all features contribute equally.
```

### Scaling Methods

| Method | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| **MinMax** | (x - min) / (max - min) | [0, 1] | Default for NN |
| **Standard** | (x - mean) / std | ~[-3, 3] | If normal dist |

### ⚠️ DATA LEAKAGE WARNING ⚠️

```python
# ❌ WRONG - Uses test data statistics!
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Sees ALL data including test
X_train, X_test = X_scaled[:split], X_scaled[split:]

# ✅ CORRECT - Only fit on training data!
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X[:split])   # Fit ONLY on train
X_test = scaler.transform(X[split:])        # Transform test with train stats
```

```
┌────────────────────────────────────────────────────────────┐
│                    DATA LEAKAGE                             │
│                                                             │
│   Train Data ─────────────────┬────────── Test Data        │
│                               │                             │
│   fit_transform()             │    transform() ONLY!       │
│   Learn min, max, mean, std   │    Use train's stats       │
│                               │                             │
│                      WALL ────┴────                        │
│                (never cross this!)                          │
└────────────────────────────────────────────────────────────┘
```

---

# Part 2: Neural Network Fundamentals (Day 10)

## 2.1 What is a Neuron?

```
         Input Weights            Activation
            │                        │
    x₁ ──── w₁ ──┐                  │
                 │                   │
    x₂ ──── w₂ ──┼──► Σ + b ──► f(z) ──► output
                 │    ▲
    x₃ ──── w₃ ──┘    │
                      │
                   bias b

z = w₁x₁ + w₂x₂ + w₃x₃ + b  (weighted sum)
output = f(z)                (activation function)
```

## 2.2 Activation Functions

```
SIGMOID (for probabilities):
1 │     _______________
  │    /
  │   /
0 │__/
  └─────────────────────
    -4   0   4

ReLU (most common in hidden layers):
  │      /
  │     /
  │    /
0 │___/
  └─────────────────────
    -2  0  2  4

TANH (for values between -1 and 1):
 1│     _______________
  │    /
0 │---/---
  │  /
-1│_/
  └─────────────────────
```

## 2.3 The Vanishing Gradient Problem

```
DEEP NETWORK:
Input → Layer1 → Layer2 → Layer3 → ... → Layer10 → Output
                                              │
                                         Error = 0.5
                                              │
Backpropagation (chain rule):                │
                                              ▼
Layer10 gradient: 0.5 × 0.25 = 0.125
Layer9 gradient:  0.125 × 0.25 = 0.03
Layer8 gradient:  0.03 × 0.25 = 0.008
...
Layer1 gradient:  ~0.0000001  ← Almost zero! Can't learn!

This is the VANISHING GRADIENT problem with sigmoid/tanh.
Deep networks can't train properly.
```

## 2.4 Why RNNs Struggle

```
RNN Processing Sequence:
                                    
  x₁ ──► [h₁] ──► x₂ ──► [h₂] ──► x₃ ──► [h₃] ──► ... ──► x₃₀ ──► [h₃₀]
          │               │               │                         │
          └───────────────┴───────────────┴─────────────────────────┘
                        Same weights shared!

Problem: To learn from x₁, gradient must flow through 30 steps.
Each step multiplies gradient by small number → vanishes to 0.

Result: RNN "forgets" early inputs!
```

---

# Part 3: LSTM Deep Dive (Day 10-11)

## 3.1 LSTM Architecture

```
                        ┌───────────────────────────────────────┐
                        │              LSTM CELL                 │
                        │                                        │
 Cell State ─────────────────────────────────────────────────────────►
 (Long-term memory)     │    ×         +                        │
                        │    │         │                        │
                        │  ┌─┴─┐    ┌──┴──┐     ┌─────┐        │
                        │  │ f │    │  i  │  ×  │  C̃  │        │
                        │  │   │    │     │     │     │        │
                        │  └───┘    └─────┘     └─────┘        │
                        │  Forget    Input      Candidate       │
                        │   Gate      Gate       Memory         │
                        │                                        │
                        │           ┌─────┐                      │
                        │           │  o  │  ×  tanh(C)         │
                        │           │     │                      │
                        │           └─────┘                      │
                        │           Output                       │
                        │            Gate                        │
                        │                                        │
 Hidden State ◄──────────────────────────────────────────────────
 (Short-term memory)    │                                        │
                        └───────────────────────────────────────┘
                                      ▲     ▲
                                      │     │
                                     x_t   h_{t-1}
```

## 3.2 The Three Gates Explained

### 1. Forget Gate: What to throw away?

```python
f = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
# Output: values between 0 and 1
# 0 = forget completely
# 1 = keep completely
```

Example: Reading text "The cat, which was black, sat..."
When we see "sat", we might forget some details about "black" since we now know the action.

### 2. Input Gate: What new info to store?

```python
i = sigmoid(W_i · [h_{t-1}, x_t] + b_i)    # What to update?
C̃ = tanh(W_C · [h_{t-1}, x_t] + b_C)       # New candidate values
```

Example: When we see "The stock split 2:1", we create new memory about the split.

### 3. Output Gate: What to output?

```python
o = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
h_t = o × tanh(C_t)
```

Example: The cell state contains all info, but we only output what's relevant for next prediction.

## 3.3 Why LSTM Solves Vanishing Gradients

```
REGULAR RNN:
Gradient path: × × × × × × × ... (30 multiplications)
Result: 0.5^30 ≈ 0  (vanishes!)

LSTM:
Gradient path through cell state: + + + + + ... (additions!)
Result: Gradients can flow unchanged!

The CELL STATE acts like a "highway" for gradients.
Gates control what goes in/out, but don't block the highway.
```

## 3.4 Building LSTM in Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Build LSTM model for time series prediction.
    
    Args:
        input_shape: (timesteps, features) e.g., (30, 5)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # LSTM layer
        LSTM(
            units=64,              # Number of LSTM neurons
            input_shape=input_shape,
            return_sequences=False  # Only return final output
        ),
        
        # Regularization
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dense(1)  # Output: predicted log return (regression)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Usage:
model = build_lstm_model(input_shape=(30, 5))
model.summary()
```

### Model Summary Output

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm (LSTM)                 (None, 64)                17920    
dropout (Dropout)           (None, 64)                0         
dense (Dense)               (None, 32)                2080      
dense_1 (Dense)             (None, 1)                 33        
=================================================================
Total params: 20,033
Trainable params: 20,033
```

## 3.5 Key Hyperparameters

| Parameter | Our Value | Meaning |
|-----------|-----------|---------|
| `units` | 64 | Number of LSTM neurons |
| `dropout` | 0.2 | 20% of neurons randomly disabled |
| `return_sequences` | False | Only output at end of sequence |
| `learning_rate` | 0.001 | Step size for gradient descent |
| `batch_size` | 32 | Samples per gradient update |
| `epochs` | 100 | Full passes through training data |

---

# Part 4: Training Infrastructure (Day 11)

## 4.1 Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',     # Watch validation loss
    patience=10,            # Stop after 10 epochs without improvement
    restore_best_weights=True  # Use best weights, not last weights
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[early_stop]
)
```

```
Epoch vs Loss:
Loss │
     │ ████                          ← Training loss (always drops)
     │     ████
     │         ████____████████████  ← Validation loss (plateaus then rises)
     │                     │
     │                     └── STOP HERE! (early stopping)
     └──────────────────────────────
       0    20    40    60    80  Epochs
```

## 4.2 Logging (Not Print!)

```python
# ❌ BAD - print statements scattered everywhere
print(f"Epoch {epoch}: loss = {loss}")

# ✅ GOOD - proper logging
import logging

logger = logging.getLogger(__name__)
logger.info(f"Epoch {epoch}: loss = {loss:.4f}")
```

Your `utils/logger.py` is already set up! Use it:

```python
from utils.logger import logger

logger.info("Starting LSTM training...")
logger.debug(f"Input shape: {X_train.shape}")
logger.warning("Validation loss increasing - possible overfitting")
logger.error("Training failed: GPU out of memory")
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"How do you prevent data leakage when scaling time series?"**
   - Fit scaler on training data ONLY
   - Transform test data using training statistics
   - In walk-forward: refit scaler for each fold

2. **"What's the difference between `fit_transform()` and `transform()`?"**
   - `fit_transform()`: Learn parameters AND apply transformation
   - `transform()`: Apply transformation using already-learned parameters

3. **"Why use logging instead of print statements?"**
   - Configurable levels (DEBUG, INFO, WARNING, ERROR)
   - Can output to file AND console
   - Can be turned off in production
   - Includes timestamps automatically

4. **"How would you save and load a Keras model?"**
   ```python
   model.save('model.h5')           # Save
   model = tf.keras.models.load_model('model.h5')  # Load
   ```

## ML-Focused Questions

1. **"Explain the vanishing gradient problem and how LSTM solves it."**
   - Gradients get multiplied through many layers → become tiny
   - LSTM has a cell state with additive updates
   - Gradients flow through cell state without shrinking
   - Gates control information flow without blocking gradients

2. **"What do the three LSTM gates do?"**
   - Forget Gate: Decides what to remove from cell state
   - Input Gate: Decides what new info to add
   - Output Gate: Decides what to output as hidden state

3. **"Why use Dropout in neural networks?"**
   - Regularization technique
   - Randomly "turns off" neurons during training
   - Prevents co-adaptation of neurons
   - Reduces overfitting

4. **"What's the difference between `return_sequences=True/False`?"**
   - `False`: Return only the final hidden state (for single prediction)
   - `True`: Return hidden state at every timestep (for sequence-to-sequence)

5. **"Why MSE loss for regression instead of accuracy?"**
   - Accuracy is for classification (right/wrong)
   - MSE penalizes predictions proportionally to error size
   - Works with gradient descent (differentiable)

---

# ✅ Day 8-11 Checklist

## Day 8 Deliverables
- [ ] Watch 3Blue1Brown neural network videos
- [ ] Create `windowing.py`
- [ ] Implement `create_windows()` function
- [ ] Test shape is correct: (samples, window, features)

## Day 9 Deliverables
- [ ] Create `tests/test_windowing.py`
- [ ] Test output shapes
- [ ] Implement feature scaling WITHOUT leakage
- [ ] Test that scaler only uses train data

## Day 10 Deliverables
- [ ] Watch StatQuest RNN and LSTM videos
- [ ] Read Colah's LSTM blog (crucial!)
- [ ] Create `models/lstm_model.py`
- [ ] Implement `build_lstm_model()` function

## Day 11 Deliverables
- [ ] Add Early Stopping callback
- [ ] Verify logging works properly
- [ ] Add model save/load utilities
- [ ] Test model builds without errors

---

# 📖 Quick Reference

```python
# Windowing
def create_windows(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # target column
    return np.array(X), np.array(y)

# Scaling (NO LEAKAGE!)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Only transform!

# LSTM Model
model = Sequential([
    LSTM(64, input_shape=(30, 5)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X, y, callbacks=[early_stop])
```

---

Now go watch those videos and implement! The LSTM gate animations in StatQuest are 🔥! 🚀
