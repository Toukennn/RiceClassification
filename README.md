# 🍚 Rice Grain Classification (Neural Network + Streamlit)

A complete end-to-end tabular machine learning project for binary classification of rice varieties using a PyTorch neural network and deployed with Streamlit.

---

## 🖥️ Streamlit Demo

<p align="center">
  <img src="assets/demo.png" width="900">
</p>

## 📌 Project Overview

This project predicts the rice variety (**Cammeo** vs **Osmancik**) using morphological features such as:

- Area  
- MajorAxisLength  
- MinorAxisLength  
- Eccentricity  
- ConvexArea  
- EquivDiameter  
- Extent  
- Perimeter  
- Roundness  
- AspectRatio  

The pipeline includes:

- Stratified train/validation/test split (70/15/15)
- Feature scaling using **MaxAbsScaler**
- Feedforward neural network (PyTorch)
- Proper evaluation with classification metrics
- Interactive Streamlit web app

---

## 🧠 Model Architecture

A simple yet effective fully connected neural network:

Input → Linear → ReLU → Linear → Output (logits)

- Hidden layer size: 32 neurons  
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr = 1e-3)

---

## 📊 Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **98.97%** |
| Precision | 0.989 |
| Recall | 0.992 |
| F1-score | 0.990 |
| Errors | 28 / 2728 |


The model demonstrates strong separability and balanced performance across both classes.

---

## 🛠️ Project Structure

The model demonstrates strong separability and balanced performance across both classes.
```text
Rice-Classification/
│
├── app.py
├── requirements.txt
├── rice_model.pt
├── maxabs_scaler.joblib
├── metadata.joblib
├── Tabular_classification.ipynb
├── README.md
└── .gitignore
```

---

## 🚀 Running the Streamlit App

1️⃣ Clone the repository:
```bash
git clone https://github.com/Toukennn/RiceClassification.git
cd RiceClassification
```
2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run the Streamlit app
```bash
streamlit run house_prediction.py
```


