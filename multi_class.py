import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report, confusion_matrix, roc_curve, auc
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import resample
from helpers import *
st.set_page_config(page_title="Skin Cancer Prediction", layout="wide")

st.title("Skin Cancer Detection")
st.write("Dự đoán ung thư da dựa trên metadata bệnh nhân")
st.header("Dataset")
data = pd.read_csv("data/HAM10000_metadata.csv")

st.dataframe(data)


with st.expander("Overview Data"):
    st.subheader("Data Type")
    st.write(data.dtypes)
    st.subheader("Mô tả dữ liệu")  
    st.write(data.describe())
    st.subheader("Missing Value")
    st.write(data.isnull().sum())
    
st.header("Data sau khi đã xử lý dữ liệu thiếu")

data_no_null = data.copy()
# imputer = SimpleImputer(strategy="most_frequent")
# data_no_null[["age", "sex", "localization"]] = imputer.fit_transform(data_no_null[["age", "sex", "localization"]])

data_no_null["age"].fillna(data_no_null["age"].mode(),inplace=True)

st.dataframe(data_no_null)
with st.expander("Overview Data"):
    st.subheader("Data Type")
    st.write(data_no_null.dtypes)
    st.subheader("Mô tả dữ liệu")  
    st.write(data_no_null.describe())
    st.subheader("Missing Value")
    st.write(data_no_null.isnull().sum())
    
# EDA 
st.header("EDA")
plot_all_histograms(data_no_null)

# Chuẩn hóa dữ liệu 
st.header("Chuẩn hóa dữ liệu")
data_no_null = data_no_null.drop(columns=["lesion_id", "image_id"])
data_no_null["sex"] = data_no_null["sex"].map({"male": 1, "female": 0})
le = LabelEncoder()
le_target = LabelEncoder()
data_no_null["diagnosis_multi"] = le_target.fit_transform(data_no_null["dx"])

# {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'healthy', 5: 'mel', 6: 'nv', 7: 'vasc'}
cat_cols = ["dx", "dx_type", "localization", "dataset"]

for col in cat_cols:
    data_no_null[col] = le.fit_transform(data_no_null[col])

data_no_null["diagnosis_multi"] = le.fit_transform(data_no_null["dx"])
print(data_no_null["diagnosis_multi"].value_counts())
mapping = dict(zip(range(len(le_target.classes_)), le_target.classes_))
print(mapping)

st.subheader("Data đã được chuẩn hóa")
st.dataframe(data_no_null)
with st.expander("Overview Data"):
    st.subheader("Data Type")
    st.write(data_no_null.dtypes)
    st.subheader("Mô tả dữ liệu")  
    st.write(data_no_null.describe())


# Ma trận tương quan 
st.subheader("Ma trận tương quan")
plot_correlation_matrix(data_no_null)
st.subheader("Nhận xét: ")
st.write("""
    dx có tương quan hoàn hảo với biến mục tiêu diagnosis_multi, do đó cần loại bỏ để tránh hiện tượng rò rỉ dữ liệu (data leakage). 
    Biến age và dataset có mức tương quan trung bình với biến mục tiêu, cho thấy chúng có khả năng hỗ trợ dự đoán. 
    Trong khi đó, các biến sex và localization có tương quan thấp, 
    tuy nhiên vẫn có thể được giữ lại vì các mô hình học máy phi tuyến có thể khai thác được các mối quan hệ phức tạp giữa các đặc trưng này với nhãn bệnh.
         """)
# downsample
balanced_data = downsample_multiclass(data_no_null, "diagnosis_multi")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Before Downsample")
    plot_multiclass_distribution(data_no_null, "diagnosis_multi")
with col2: 
    st.subheader("After Downsample")
    plot_multiclass_distribution(balanced_data, "diagnosis_multi")

# Chia tập huấn luyện 
st.header("Data splitting")

st.write("""
         - Các đặc trưng cần lấy là age, sex, localization đại diện cho thông tin nhân khẩu học và lâm sàng cơ bản thường có trong môi trường y tế trong thế giới thực.
         Trong khi tuổi tác cho thấy mối tương quan mạnh nhất với biến mục tiêu, giới tính và khu vực địa phương được giữ lại như những đặc điểm bổ sung do mối liên quan về dịch tễ học và lâm sàng của chúng.
         Hơn nữa, việc sử dụng bộ phân loại phi tuyến tính (SVM với nhân RBF) cho phép mô hình nắm bắt được các tương tác phức tạp giữa các tính năng có thể không thể hiện rõ chỉ thông qua phân tích tương quan tuyến tính.
         - Các biến như dx và dx_type không được sử dụng làm đặc trưng đầu vào do có nguy cơ gây ra hiện tượng data leakage, làm sai lệch kết quả đánh giá mô hình
         """)

feature = ["age","sex", "localization", "dataset"]
X = balanced_data[feature]
y = balanced_data["diagnosis_multi"]


st.header("Quy trình chuẩn hóa dữ liệu")
st.write(f"""
         - Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
         - Xử lý dữ liệu thiếu bằng cách sử dụng SimpleImputer với chiến lược most_frequent
         - Chuẩn hóa dữ liệu bằng StandardScaler
         """)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2,random_state=42)
imputer = SimpleImputer(strategy="most_frequent")
X_train_impute = imputer.fit_transform(X_train)
X_test_impute = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_impute)
X_test_scaled = scaler.transform(X_test_impute)

# Model
st.header("Xây dựng mô hình SVM")
svm_multi = SVC(kernel='rbf', C=1, gamma='scale',probability= True, random_state=42)
svm_multi.fit(X_train_scaled, y_train)
# st.write("Mô hình SVM đã được huấn luyện thành công.")


y_pred = svm_multi.predict(X_test_scaled)
# Đánh giá mô hình
st.subheader("Đánh giá mô hình với SVM")
y_pred = svm_multi.predict(X_test_scaled)
st.write(f"Accuracy {accuracy_score(y_test,y_pred,)}")
st.write(f"Precision {precision_score(y_test,y_pred, average='macro')}")
st.write(f"Recall {recall_score(y_test,y_pred, average='macro')}")
st.write(f"F1-score {f1_score(y_test,y_pred, average='macro')}")

st.write(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

# Chuyển nhãn số về nhãn chữ để hiển thị ma trận nhầm lẫn

y_test_label = le.inverse_transform(y_test)
y_pred_label = le.inverse_transform(y_pred)

cm = confusion_matrix(y_test_label, y_pred_label)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap="Blues",
        ax = ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)
plt.close(fig)

# ROC Curve
st.subheader("ROC Curve") 

plot_multiclass_roc_auc_streamlit(svm_multi, X_test_scaled, y_test)

st.write ("""
          - Nhận xét: 
          Mô hình hoạt động tốt trên các lớp healthy, akiec, nv nhưng còn hạn chế trong việc phân biệt bkl, mel, vasc. Tổng thể mô hình đạt hiệu quả khá với AUC trung bình khoảng 0.77 – 0.80.
          """)
st.subheader("Decision Boundary sau khi giảm chiều dữ liệu với PCA")
# Apply PCA to reduce the data to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
x_train_pca = pca.fit_transform(X_train_scaled)
x_test_pca = pca.transform(X_test_scaled)

# Train a new SVM model on the PCA-transformed data
svm_model_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model_pca.fit(x_train_pca, y_train)

plot_decision_boundary(x_test_pca, y_test, svm_model_pca)

# Predict new data
st.sidebar.header("Predict New Data")
# Load encoder
class_mapping = dict(zip(range(len(le_target.classes_)), le_target.classes_))
# input
age_input = st.sidebar.number_input("Age", min_value=0, max_value=120, value=40)
sex_input = st.sidebar.selectbox("Sex", ["male", "female"])
localization_input = st.sidebar.selectbox("Localization",options=data["localization"].dropna().unique())
dataset_input = st.sidebar.selectbox("Dataset",options=data["dataset"].dropna().unique())
sex_encoded = 1 if sex_input == "male" else 0
localization_encoded = le.fit(data["localization"]).transform([localization_input])[0]
dataset_encoded = le.fit(data["dataset"]).transform([dataset_input])[0]

input_df = pd.DataFrame({
    "age": [age_input],
    "sex": [sex_encoded],
    "localization": [localization_encoded],
    "dataset": [dataset_encoded]})


input_impute = imputer.transform(input_df)
input_scaled = scaler.transform(input_impute)
disease_description = {
    "akiec": "Dày sừng ánh sáng (có thể tiến triển thành ung thư da)",
    "bcc": "Ung thư biểu mô tế bào đáy",
    "bkl": "Tổn thương da lành tính (dày sừng, lentigo...)",
    "df": "U xơ da (lành tính)",
    "healthy": "Da bình thường",
    "mel": "Ung thư hắc tố (Melanoma - nguy hiểm)",
    "nv": "Nốt ruồi sắc tố (lành tính)",
    "vasc": "Tổn thương mạch máu"
}
if st.sidebar.button("Predict"):
    pred = svm_multi.predict(input_scaled)[0]
    pred_label = le_target.inverse_transform([pred])[0]
    #mô tả bệnh
    disease_text = disease_description[pred_label]
    st.sidebar.success(f"Prediction: {pred_label}")
    st.sidebar.info(f"Description: {disease_text}")
    if pred_label == "mel":
        st.sidebar.error("Cảnh báo: Có dấu hiệu Melanoma")
