# 📚 Phase 3 Notes (Part 1): Days 15-18
## NLP Fundamentals + Attention Mechanism

> **Duration:** ~10-12 hours total  
> **Goal:** Add sentiment analysis and attention layer to model  
> **Output:** `sentiment.py`, `models/attention.py`, `models/attention_lstm.py`

---

# 🗺️ Learning Roadmap

```
Day 15: NLP Basics
├── Text preprocessing
├── Tokenization
├── Why sentiment matters for stocks

Day 16: VADER Sentiment + News API
├── VADER sentiment scoring
├── NewsAPI integration
├── Aggregating daily sentiment

Day 17: Attention Mechanism Theory
├── Why attention?
├── Key, Query, Value concept
├── Self-attention basics

Day 18: Multi-Input Model
├── Custom Keras layers
├── Functional API
├── Dual-branch architecture
```

---

# 📺 Videos to Watch FIRST

| Order | Topic | Video | Duration | Priority |
|-------|-------|-------|----------|----------|
| 1 | NLP Basics | [NLP Zero to Hero Part 1](https://youtu.be/fNxaJsNG3-s) | 15 min | 🟡 Good |
| 2 | VADER Sentiment | [VADER Sentiment Analysis](https://youtu.be/xvqsFTUsOmc) | 15 min | 🔴 Must |
| 3 | Attention Intuition | [Attention in 10 min](https://youtu.be/fjJOgb-E41w) | 10 min | 🔴 Must |
| 4 | Attention Visual | [Jay Alammar - Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) | 20 min | 🔴 Must |
| 5 | Keras Functional | [Keras Functional API](https://youtu.be/EvGS3VAsG4Y) | 15 min | 🔴 Must |

**Additional (Highly Recommended):**
- [Attention Is All You Need (Paper Explained)](https://youtu.be/iDulhoQ2pro) - For deep understanding

---

# Part 1: NLP Fundamentals (Day 15)

## 1.1 Why Sentiment for Stock Prediction?

```
┌─────────────────────────────────────────────────────────────┐
│                 MARKET INFLUENCES                            │
│                                                              │
│   Technical Signals ────────────────┬──────► Price Change   │
│   (RSI, MACD, etc.)                 │                        │
│                                      │                        │
│   Sentiment Signals ────────────────┘                        │
│   (News, Social Media)                                       │
│                                                              │
│   "Apple announces record earnings" → Positive sentiment    │
│   "Fed raises interest rates"       → Negative sentiment    │
│   "CEO resigns unexpectedly"        → Negative sentiment    │
│                                                              │
│   Technical alone: 55% accuracy                              │
│   Technical + Sentiment: 57-60% accuracy (goal)             │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 NLP Pipeline Overview

```
RAW TEXT                           PROCESSED
─────────────────────────────────────────────────────────
"Apple's new iPhone is          → Tokens: ['apple', 'new', 
 absolutely amazing!!!"            'iphone', 'absolutely', 
                                   'amazing']
                                   
                                → Sentiment: +0.85 (very positive)
                                
"Stock market faces             → Tokens: ['stock', 'market',
 major concerns today"            'faces', 'major', 'concerns']
                                   
                                → Sentiment: -0.45 (negative)
```

## 1.3 Key NLP Concepts

### Tokenization
Breaking text into individual words/tokens.

```python
text = "Apple's stock rose 5% today!"

# Basic split (bad)
tokens = text.split()  # ["Apple's", "stock", "rose", "5%", "today!"]

# NLTK tokenization (better)
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)  # ["Apple", "'s", "stock", "rose", "5", "%", "today", "!"]
```

### Stopwords
Common words that don't carry meaning.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ["the", "is", "at", "which", "on", "a", "an", ...]
```

### Stemming vs Lemmatization

```
Stemming (crude, fast):
  "running" → "run"
  "better"  → "bet" (wrong!)
  
Lemmatization (accurate, slower):
  "running" → "run"
  "better"  → "good" (correct!)
```

---

# Part 2: VADER Sentiment (Day 16)

## 2.1 What is VADER?

**V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner

- Rule-based sentiment analyzer
- Works great for social media/news
- No training required
- Handles emojis, slang, punctuation

## 2.2 VADER Scores Explained

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores("Apple stock is doing great!")

# Output:
{
    'neg': 0.0,      # Negative score (0-1)
    'neu': 0.513,    # Neutral score (0-1)
    'pos': 0.487,    # Positive score (0-1)
    'compound': 0.6249  # Overall sentiment (-1 to +1)
}
```

### Compound Score Interpretation

```
Compound Score Range:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-1.0          -0.05    0.05           1.0
 │              │        │              │
 ▼              ▼        ▼              ▼
Negative       Neutral Zone         Positive

Examples:
"Apple crashed badly"      → compound = -0.75
"Stock went up slightly"   → compound = +0.15
"Market is mixed"          → compound = +0.02 (neutral)
```

## 2.3 VADER Special Handling

```
VADER is smart! It handles:

1. Capitalization: "GREAT" more intense than "great"
2. Punctuation: "great!!!" more intense than "great"
3. Modifiers: "extremely good" > "good"
4. Negation: "not good" → negative
5. Emojis: "amazing 🔥" → more positive
```

## 2.4 NewsAPI Integration

```python
import requests
from datetime import datetime, timedelta

def fetch_news(ticker: str, date: str, api_key: str) -> list:
    """
    Fetch news articles for a ticker on a specific date.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        date: Date in "YYYY-MM-DD" format
        api_key: NewsAPI key
    
    Returns:
        List of article texts
    """
    base_url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": ticker,
        "from": date,
        "to": date,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    articles = []
    for article in data.get("articles", []):
        text = f"{article['title']} {article['description']}"
        articles.append(text)
    
    return articles
```

> [!WARNING]
> NewsAPI free tier has limitations:
> - 100 requests/day
> - No access to articles older than 30 days
> - Consider caching results!

## 2.5 Aggregating Daily Sentiment

```python
def compute_daily_sentiment(articles: list) -> dict:
    """
    Compute aggregated sentiment for a day's articles.
    
    Returns:
        Dict with 'compound_mean', 'sentiment_std', 'news_count'
    """
    if not articles:
        return {
            'compound_mean': 0,
            'sentiment_std': 0,
            'news_count': 0
        }
    
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(text)['compound'] for text in articles]
    
    return {
        'compound_mean': np.mean(scores),
        'sentiment_std': np.std(scores),
        'news_count': len(articles)
    }
```

### Feature Engineering from Sentiment

| Feature | Meaning | Why Useful |
|---------|---------|------------|
| `compound_mean` | Average daily sentiment | Main sentiment signal |
| `sentiment_std` | Sentiment volatility | Disagreement in news |
| `news_count` | Number of articles | High count = important day |

## 2.6 Avoiding Data Leakage with Sentiment

```
⚠️ CRITICAL: Use sentiment(t) to predict returns(t+1)

WRONG (leakage):
  sentiment(t) → predict returns(t)
  (News published DURING the day affects same day's close)

CORRECT:
  sentiment(t) → predict returns(t+1)
  (News from today affects tomorrow's price)
```

---

# Part 3: Attention Mechanism (Day 17)

## 3.1 Why Attention?

```
LSTM Problem:
Input: [Day1, Day2, Day3, ..., Day28, Day29, Day30]
                                           │
                                           ▼
                                    Final Hidden State
                                    (Has to remember 
                                     everything!)

With Attention:
Input: [Day1, Day2, Day3, ..., Day28, Day29, Day30]
         │      │      │              │      │      │
         └──────┼──────┼──────────────┼──────┼──────┘
                       │
                       ▼
              Weighted combination!
              "Day 5 matters most for 
               today's prediction"
```

## 3.2 Attention Intuition

```
Question: "What day should I pay most attention to?"

Each Day has:
  - Key (K): "What information do I have?"
  - Value (V): "What's my actual content?"

Current State has:
  - Query (Q): "What am I looking for?"

Attention = softmax(Q · K^T) × V

In plain English:
1. Compare query with all keys (dot product)
2. Normalize to get weights (softmax)
3. Weight the values by importance
4. Sum to get context vector
```

## 3.3 Attention Step by Step

```
Example: 3-day window

Hidden States from LSTM:
  h₁ = [0.1, 0.2, 0.3, 0.4]  (Day 1 encoding)
  h₂ = [0.5, 0.3, 0.2, 0.1]  (Day 2 encoding)
  h₃ = [0.3, 0.4, 0.5, 0.2]  (Day 3 encoding)

Step 1: Compute Attention Scores
  score₁ = tanh(W · h₁ + b)  →  1.2
  score₂ = tanh(W · h₂ + b)  →  0.8
  score₃ = tanh(W · h₃ + b)  →  2.1

Step 2: Softmax to get Weights
  α₁ = exp(1.2) / Σexp  →  0.21
  α₂ = exp(0.8) / Σexp  →  0.14
  α₃ = exp(2.1) / Σexp  →  0.65  ← Day 3 most important!

Step 3: Weighted Sum (Context Vector)
  context = 0.21·h₁ + 0.14·h₂ + 0.65·h₃
          = [0.28, 0.36, 0.42, 0.22]
```

## 3.4 Attention Visualization

```
Attention Weights Heatmap:

        Day 1  Day 2  Day 3  Day 4  Day 5
         │      │      │      │      │
         ▼      ▼      ▼      ▼      ▼
Sample1: ░░░   ░░░░  ███████  ░░    ░░░░
Sample2: ░░░░░ ░░    ░░     ░░░░  █████████
Sample3: ██████ ░░░   ░░░    ░░░   ░░░
         │                          │
      Light = Low attention    Dark = High attention
      
Interpretation:
- Sample1 focuses on Day 3 (maybe big news that day)
- Sample2 focuses on Day 5 (recent momentum)
- Sample3 focuses on Day 1 (longer-term pattern)
```

## 3.5 Custom Attention Layer in Keras

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    """
    Bahdanau-style attention layer for time series.
    
    Computes attention weights and returns context vector.
    """
    
    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="attention_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        
        # Score = tanh(inputs @ W + b) @ u
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), 
            axis=1
        )
        
        # Context = weighted sum of inputs
        context = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
```

---

# Part 4: Multi-Input Model (Day 18)

## 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-INPUT MODEL                         │
│                                                              │
│   Price Features (30 days)         Sentiment Features        │
│   [Open, Close, RSI, MACD, Vol]    [compound, std, count]   │
│            │                              │                  │
│            ▼                              ▼                  │
│   ┌─────────────────┐            ┌───────────────┐          │
│   │ LSTM (64 units) │            │ Dense (16)    │          │
│   └────────┬────────┘            │ Dense (8)     │          │
│            │                      └───────┬───────┘          │
│            ▼                              │                  │
│   ┌─────────────────┐                    │                  │
│   │ Attention Layer │                    │                  │
│   └────────┬────────┘                    │                  │
│            │                              │                  │
│            └──────────┬──────────────────┘                  │
│                       │                                      │
│                       ▼                                      │
│              ┌────────────────┐                             │
│              │  Concatenate   │                             │
│              └───────┬────────┘                             │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                             │
│              │  Dense (32)    │                             │
│              │  Dense (1)     │ ────► Predicted Return       │
│              └────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Keras Functional API

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate

def build_attention_lstm_model(
    price_input_shape: tuple,      # (30, 5) - 30 days, 5 features
    sentiment_input_shape: tuple   # (3,) - 3 sentiment features
) -> Model:
    """Build multi-input model with attention."""
    
    # Input 1: Price features (time series)
    price_input = Input(shape=price_input_shape, name="price_input")
    
    # LSTM branch
    lstm_out = LSTM(64, return_sequences=True)(price_input)
    
    # Attention layer
    attention_layer = AttentionLayer(units=32)
    context, attention_weights = attention_layer(lstm_out)
    
    # Dropout for regularization
    context = Dropout(0.2)(context)
    
    # Input 2: Sentiment features (static)
    sentiment_input = Input(shape=sentiment_input_shape, name="sentiment_input")
    
    # Dense layers for sentiment
    sentiment_dense = Dense(16, activation='relu')(sentiment_input)
    sentiment_dense = Dense(8, activation='relu')(sentiment_dense)
    
    # Merge branches
    merged = Concatenate()([context, sentiment_dense])
    
    # Final dense layers
    dense = Dense(32, activation='relu')(merged)
    dense = Dropout(0.2)(dense)
    output = Dense(1, name="output")(dense)
    
    # Build model
    model = Model(
        inputs=[price_input, sentiment_input],
        outputs=output
    )
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
```

## 4.3 Training Multi-Input Model

```python
# Prepare inputs
X_price = price_windows       # Shape: (samples, 30, 5)
X_sentiment = sentiment_data  # Shape: (samples, 3)
y = target_returns            # Shape: (samples,)

# Build model
model = build_attention_lstm_model(
    price_input_shape=(30, 5),
    sentiment_input_shape=(3,)
)

# Train with multiple inputs
model.fit(
    [X_price, X_sentiment],  # List of inputs!
    y,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# Predict
predictions = model.predict([X_price_test, X_sentiment_test])
```

## 4.4 Extracting Attention Weights

```python
# Create a model that outputs attention weights
attention_model = Model(
    inputs=model.input,
    outputs=attention_layer.output[1]  # Second output is weights
)

# Get weights for a sample
weights = attention_model.predict([X_price[:1], X_sentiment[:1]])
# Shape: (1, 30, 1) - weight for each of 30 days

# Visualize
import matplotlib.pyplot as plt
plt.bar(range(30), weights[0].flatten())
plt.xlabel('Day')
plt.ylabel('Attention Weight')
plt.title('Which days did the model focus on?')
```

---

# 🎯 Interview Questions

## SDE-Focused Questions

1. **"How do you handle rate limits when calling external APIs?"**
   - Implement retry with exponential backoff
   - Cache results locally
   - Use bulk endpoints when available
   - Respect rate limit headers

2. **"What's the difference between Sequential and Functional API in Keras?"**
   - Sequential: Linear stack of layers, simple
   - Functional: Flexible graph structure, multiple inputs/outputs
   - Use Functional for complex architectures

3. **"How would you test a custom Keras layer?"**
   ```python
   def test_attention_output_shape():
       layer = AttentionLayer(units=32)
       input_tensor = tf.random.normal((2, 30, 64))
       context, weights = layer(input_tensor)
       assert context.shape == (2, 64)
       assert weights.shape == (2, 30, 1)
   ```

## ML-Focused Questions

1. **"Explain the attention mechanism in simple terms."**
   - Allows model to focus on relevant parts of input
   - Computes importance weights for each timestep
   - Creates weighted combination instead of using just final state
   - Makes model more interpretable

2. **"What's the difference between self-attention and cross-attention?"**
   - Self-attention: Query, Key, Value all from same sequence
   - Cross-attention: Query from one sequence, Key/Value from another
   - We use a simplified version for time series

3. **"Why combine price and sentiment features in separate branches?"**
   - Different data modalities (time series vs static)
   - Each branch can learn appropriate representations
   - Merged features capture complementary information

4. **"How do you interpret attention weights for trading?"**
   - High weight on specific day = that day's pattern is important
   - Can identify which historical days influence prediction
   - Useful for debugging and explaining model decisions

5. **"What's VADER's advantage over training a sentiment model?"**
   - No labeled data needed
   - Works out of the box
   - Handles finance-specific terms reasonably
   - Fast inference
   - Good baseline before more complex models

---

# ✅ Day 15-18 Checklist

## Day 15 Deliverables
- [ ] Read NLP basics materials
- [ ] Install NLTK and download resources
- [ ] Understand tokenization concepts
- [ ] Test VADER on sample sentences

## Day 16 Deliverables
- [ ] Create `sentiment.py`
- [ ] Implement NewsAPI fetching (or use alternative)
- [ ] Implement VADER scoring function
- [ ] Implement daily aggregation function
- [ ] Merge sentiment with price data
- [ ] Create `tests/test_sentiment.py`

## Day 17 Deliverables
- [ ] Watch attention mechanism videos
- [ ] Read Jay Alammar's attention blog
- [ ] Understand Q, K, V concepts
- [ ] Create `models/attention.py`
- [ ] Implement `AttentionLayer` class

## Day 18 Deliverables
- [ ] Watch Keras Functional API video
- [ ] Create `models/attention_lstm.py`
- [ ] Implement `build_attention_lstm_model()`
- [ ] Test model builds correctly
- [ ] Create `tests/test_models.py`

---

# 📖 Quick Reference

```python
# VADER Sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores(text)['compound']

# Attention Layer (simplified)
score = tanh(W @ hidden_states + b)
weights = softmax(score)
context = sum(weights * hidden_states)

# Multi-Input Model
price_input = Input(shape=(30, 5))
sentiment_input = Input(shape=(3,))
# ... build branches ...
merged = Concatenate()([branch1, branch2])
model = Model(inputs=[price_input, sentiment_input], outputs=output)

# Train with multiple inputs
model.fit([X_price, X_sentiment], y)
```

---

Now master attention - it's the foundation of modern AI! 🚀
