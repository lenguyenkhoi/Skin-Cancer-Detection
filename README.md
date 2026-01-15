# Skin-Cancer-Detection
This project focuses on building a **machine learning model** to detect **skin cancer (melanoma vs benign)** using patient metadata from the **HAM10000 dataset**.  
The goal is to demonstrate an end-to-end ML pipeline, from data preprocessing to model evaluation and deployment with Streamlit.
---
Link app: https://skin-cancer-detection-demo.streamlit.app/
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
├── helpers.py
├── streamlit_demo.py
├── README.md
```

---

## 📑 HAM10000 Dataset Overview

The **HAM10000** dataset ("Human Against Machine with 10,000 training images") is a collection of dermatoscopic images designed for training machine learning models in **skin cancer diagnosis.**

### 1. Class Labels

The dataset is divided into 7 main classes, including both benign and malignant skin lesions:
* **Benign Lesions:**
* `nv`: **Melanocytic nevus** (Nốt ruồi bình thường).
* `bkl`: **Benign keratosis** (Dày sừng lành tính).
* `df`: **Dermatofibroma** (U xơ da lành tính).
* `vasc`: **Vascular lesions** (Tổn thương mạch máu).


* **Pre-cancerous & Malignant Lesions:**
* `akiec`: **Actinic keratoses** (Tổn thương tiền ung thư).
* `bcc`: **Basal cell carcinoma** (Ung thư biểu mô tế bào đáy).
* `mel`: **Melanoma** (Ung thư hắc tố - Nguy hiểm nhất).



### 2. Metadata Structure

The file `HAM10000_metadata.csv` provides detailed information for each image:

| Field | Description|
| --- | --- |
| `lesion_id` | ID of the lesion region. |
| `image_id` | Corresponding image file name. |
| `dx` |Diagnostic label (target variable). |
| `dx_type` | Method of diagnosis (biopsy, expert consensus, etc.). |
| `age` |Patient’s age. |
| `sex` | Patient’s gender. |
| `localization` | Anatomical site of the lesion. |

### 3. Technical Characteristics

* **Size:** 10,015 color images.
* **Challenge:** The dataset is **highly imbalanced** (the `nv` class dominates), requiring data augmentation or sampling techniques during model training.

---
