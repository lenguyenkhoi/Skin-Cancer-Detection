# Skin-Cancer-Detection
This project focuses on building a **machine learning model** to detect **skin cancer (melanoma vs benign)** using patient metadata from the **HAM10000 dataset**.  
The goal is to demonstrate an end-to-end ML pipeline, from data preprocessing to model evaluation and deployment with Streamlit.

---

## 📌 Project Overview

- **Task:** Binary classification (Melanoma vs Benign)
- **Dataset:** HAM10000 (metadata only)
- **Model:** Support Vector Machine (SVM)
- **Output:** Skin cancer prediction based on patient information

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Handle missing values with `SimpleImputer`
   - Encode categorical variables (sex, localization)
   - Feature scaling using `StandardScaler`
   - Handle class imbalance using downsampling

2. **Model Training**
   - Train an SVM classifier for binary classification
   - Split data into training and testing sets

3. **Evaluation**
   - Confusion Matrix
   - ROC Curve & ROC-AUC (≈ 0.71)

4. **Deployment**
   - Build a simple Streamlit interface for real-time prediction

---

## 📊 Results

- The model achieves a **ROC-AUC of approximately 0.71**
- Demonstrates reasonable discriminative performance using metadata features
- Suitable for **educational purposes and initial screening support**

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
streamlit run streamlit_demo.py
🔮 Future Improvements
Train a CNN model directly on dermoscopic images

Optimize SVM hyperparameters using GridSearchCV

Deploy the application to HuggingFace Spaces or Streamlit Cloud

📁 Project Structure
├── data/
├── notebooks/
├── streamlit_demo.py
├── README.md
