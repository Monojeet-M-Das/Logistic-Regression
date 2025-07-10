# Logistic Regression for Stock Direction Prediction 📉📈

This project demonstrates how to build a **logistic regression model** using historical stock price data to predict the **direction of daily returns** (up or down) using lagged features. It utilizes `scikit-learn`, `pandas`, `yfinance`, and `numpy`.

## 🚀 Overview

We train a logistic regression classifier to predict whether the price of **IBM** stock will rise or fall based on the percentage returns of the previous two days (Lag 1 and Lag 2).

## 🧠 Key Concepts

- **Lag Features**: Past day returns used as inputs
- **Target**: Binary direction label (`1` for up, `-1` for down)
- **Train/Test Split**: 70% training, 30% testing
- **Model**: Scikit-learn's `LogisticRegression`
- **Evaluation**: Accuracy score and confusion matrix

## 🛠️ Setup Instructions

Install dependencies:

```bash
pip install pandas numpy scikit-learn yfinance
```

## 📊 Output Example

Accurace of the model: 0.52
[[15 20]
 [12 18]]

This output shows:

  Model accuracy on the test set

  Confusion matrix indicating true/false positives and negatives

## 📅 Data Used

  Ticker: IBM

  Period: Jan 1, 2017 to Jan 1, 2018

  Data Source: Yahoo Finance

## ⚠️ Notes

  This is a simple demonstration and not meant for real-world trading.

  The accuracy may be close to random due to the limited and noisy nature of daily stock returns.

  Feature engineering and more advanced models (e.g., SVM, random forests, or neural networks) could improve performance.
