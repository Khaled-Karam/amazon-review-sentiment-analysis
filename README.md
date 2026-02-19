# Amazon Product Review Sentiment Analysis using RNN & LSTM

## 📌 Project Overview
This project implements Sentiment Analysis on Amazon product reviews using Recurrent Neural Networks (RNN) and LSTM architectures.

The objective is to predict product ratings (1–5 stars) based on customer review text.

---

## 📊 Dataset
Dataset: Consumer Reviews of Amazon Products  
Size: ~25,000 reviews  
Features:
- Review Text
- Rating (1–5)

---

## 🧠 Problem Statement
Given a customer review, classify it into one of 5 sentiment classes representing the rating.

This is a multi-class text classification problem.

---

## ⚙️ NLP Pipeline

1. Text Cleaning
2. Tokenization (Keras Tokenizer)
3. Sequence Conversion
4. Padding (Fixed length = 500)
5. One-Hot Encoding for Labels
6. Train-Test Split

---

## 🏗 Model Architectures

### 1️⃣ Simple RNN
- Embedding Layer
- Stacked SimpleRNN
- Dense + Softmax Output

### 2️⃣ LSTM Model
- Embedding Layer
- LSTM (150 units)
- Batch Normalization
- Dropout (0.5)
- Dense (ReLU)
- Softmax Output (5 classes)

Loss Function: `categorical_crossentropy`  
Optimizer: `Adam`


---

## 📈 Results
- LSTM outperformed Simple RNN
- Reduced overfitting using Dropout and Batch Normalization
- Achieved improved validation accuracy

---

## 🔧 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Google Colab

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

Open the notebook and run all cells.

---

## 📌 Future Improvements
- Use pretrained embeddings (GloVe / FastText)
- Replace RNN with Bidirectional LSTM
- Experiment with Transformer models (BERT)
- Add attention mechanism

---


