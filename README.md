# Graph Neural Network for Fraud Detection

## Overview
This project demonstrates how **Graph Neural Networks (GNNs)** can be applied to **fraud detection** by modeling **users and transactions as a graph**.  
Unlike traditional machine learning models that treat each transaction independently, this approach captures **relational patterns** between entities, which are common in real-world fraud scenarios.

---

## Problem Statement
Fraud detection datasets are typically **highly imbalanced** and **relational** in nature.

The objectives of this project are to:
- Detect fraudulent transactions effectively  
- Handle class imbalance correctly  
- Use appropriate evaluation metrics instead of misleading accuracy  

---

## Dataset
**IEEE-CIS Fraud Detection Dataset**

The following columns are used:
- `TransactionID`
- `card1` (used as a proxy for user identity)
- `TransactionAmt`
- `isFraud`

The full dataset is large (approximately **590k rows and 394 columns**).  
For efficiency and reproducibility, **only the required columns and a sampled subset** are loaded.

---

## Graph Construction

### Nodes
- User nodes  
- Transaction nodes  

### Edges
- A user is connected to the transactions they performed  

### Node Features
- Transaction amount  
- User nodes use dummy features  

### Labels
- Fraud labels apply **only to transaction nodes**  
- User nodes are **masked during training and evaluation**

This setup enables **node-level fraud classification** using a GNN.

---

## Model Architecture
- **Model**: Graph Convolutional Network (GCN)  
- **Framework**: PyTorch Geometric  

### Layers
- `GCNConv → ReLU → GCNConv`

### Task
- Binary node classification (fraud / non-fraud)

---

## Handling Class Imbalance
Fraud data is extremely imbalanced (approximately **3–4% fraud**).

To address this:
- **Weighted cross-entropy loss** is used  
- **Threshold tuning** is applied during evaluation  
- **Accuracy is not treated as a primary metric**

This mirrors real-world fraud detection systems where **missing fraud is more costly than false positives**.

---

## Evaluation Strategy
The model is evaluated using:
- Training vs Validation Loss  
- ROC Curve  
- Precision–Recall Curve  
- ROC AUC  
- Threshold-based Precision & Recall  
- AUC stability across multiple runs  

Accuracy is reported for completeness but is **not emphasized** due to its misleading nature on imbalanced datasets.

---

## Visualizations Included
- Training vs Validation Loss Curve  
- ROC Curve  
- Precision–Recall Curve  
- Fraud Probability Distribution  
- Sample Transaction Predictions  

---

## Results Interpretation
- Different thresholds correspond to different operating points  
- High thresholds reduce false positives but may miss fraud  
- Lower thresholds improve recall at the cost of precision  

This reflects real-world fraud detection workflows, where models are primarily used for **risk ranking and investigation**, not fully automated decisions.

---

## Limitations & Future Work
- Only transaction amount is used as a feature   
- More expressive GNNs (e.g., Graph Attention Networks) could better model neighbor importance  
- Dynamic thresholding based on business risk is possible  

---

## How to Run

### Install dependencies

-pip install -r requirements.txt
-Open the notebook - jupyter notebook fraud_detection_gnn.ipynb
-Run all cells


