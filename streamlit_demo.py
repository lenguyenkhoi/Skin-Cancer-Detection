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
data = pd.read_csv("data/Ham10000_metadata.csv")

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
data_no_null["age"].fillna(data_no_null["age"].median(),inplace=True)

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

st.header("Chuẩn hóa dữ liệu")

st.write(f"""
         Tạo thêm cột diagnosis_binary từ cột [dx]. Với tiêu chí
         - malignant = ["mel", "bcc", "akiec", "vasc"]  được gán nhãn 1
         - benign = ["nv", "df", "bkl"]                 được gán nhãn 0
         Loại bỏ cột không cần thiết lesson_id, image_id 
         Chuẩn hóa các cột ['dx', 'dx_type', 'localization','dataset'] bằng Label Encoder
         Chuẩn hóa cột sex bằng OneHotCoder(0,1)
         """)

# Chuẩn hóa data
malignant = ["mel", "bcc", "akiec", "vasc"] #1
benign = ["nv", "df", "bkl"]  #0

data_no_null = data_no_null[data_no_null["dx"].isin(malignant + benign)].copy()
data_no_null["diagnosis_binary"] = data_no_null["dx"].apply(lambda x: 1 if x in malignant else 0)

data_no_null = data_no_null.drop(columns=['lesion_id', 'image_id'])

# Chuẩn hóa cột sex thành dạng 0 1  
data_no_null['sex'] = data_no_null['sex'].map({'male': 1, 'female': 0})

cat_cols = ['dx', 'dx_type', 'localization', 'dataset']
le = LabelEncoder()
for col in cat_cols:
    data_no_null[col] = le.fit_transform(data_no_null[col])
    
st.header("Data sau khi đã chuẩn hóa")
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
    -    dx = -0.50. Tương quan âm mức trung bình–khá → dx có ảnh hưởng rõ rệt đến kết quả chẩn đoán
    -    dx_type = 0.38. Tương quan dương mức trung bình → loại chẩn đoán có giá trị dự báo
    -    age = 0.32.Tương quan dương mức trung bình → tuổi càng cao có xu hướng liên quan đến chẩn đoán dương
    -    dataset = -0.36. Tương quan âm mức trung bình → nguồn dữ liệu ảnh hưởng đến nhãn
    -    sex = 0.08. Gần 0 → ảnh hưởng rất yếu
    -    localization = -0.05. Gần 0 → hầu như không ảnh hưởng
         """)

# Imbalance

melanoma_data = data_no_null[data_no_null['diagnosis_binary'] == 0]
benign_data = data_no_null[data_no_null['diagnosis_binary'] == 1]

# Giảm melanoma = benign
df_melanoma_downsampled = melanoma_data.sample(n=len(benign_data),random_state=42)

# Data cân bằng
balanced_data = pd.concat([df_melanoma_downsampled, benign_data]).sample(frac=1,random_state=42)


# plot_class_distribution(balanced_data)

col1,col2 = st.columns(2)
with col1:
    st.subheader("Imbalance data")
    plot_class_distribution(data_no_null)
    # st.write("Before downsampling:",np.bincount(data_no_null["diagnosis_binary"]))
    st.write("Class 0:", len(melanoma_data))
    st.write("Class 1:", len(benign_data))
with col2: 
    st.subheader("Balance data")
    plot_class_distribution(balanced_data)
    # st.write("After downsampling:", np.bincount(balanced_data['diagnosis_binary']))
    st.write("Class 0:",len(balanced_data[balanced_data["diagnosis_binary"]==0]))
    st.write("Class 1:",len(balanced_data[balanced_data["diagnosis_binary"]==1]))
    
st.header("Data splitting")

st.write("""
         - Các đặc trưng cần lấy là age, sex, localization đại diện cho thông tin nhân khẩu học và lâm sàng cơ bản thường có trong môi trường y tế trong thế giới thực.
         Trong khi tuổi tác cho thấy mối tương quan mạnh nhất với biến mục tiêu, giới tính và khu vực địa phương được giữ lại như những đặc điểm bổ sung do mối liên quan về dịch tễ học và lâm sàng của chúng.
         Hơn nữa, việc sử dụng bộ phân loại phi tuyến tính (SVM với nhân RBF) cho phép mô hình nắm bắt được các tương tác phức tạp giữa các tính năng có thể không thể hiện rõ chỉ thông qua phân tích tương quan tuyến tính.
         - Các biến như dx và dx_type không được sử dụng làm đặc trưng đầu vào do có nguy cơ gây ra hiện tượng data leakage, làm sai lệch kết quả đánh giá mô hình
         """)

feature = ["age","sex", "localization"]
X = balanced_data[feature]
y= balanced_data["diagnosis_binary"]

st.header("Quy trình chuẩn hóa dữ liệu")
st.write(f"""
         - Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
         - Xử lý dữ liệu thiếu bằng cách sử dụng SimpleImputer với chiến lược most_frequent
         - Chuẩn hóa dữ liệu bằng StandardScaler
         """)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=42)

impute = SimpleImputer(strategy="most_frequent")
X_train_impute = impute.fit_transform(X_train)
X_test_impute = impute.transform(X_test)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train_impute)
X_test_scaler = scaler.transform(X_test_impute)
st.write("Kích thước tập huấn luyện:", X_train_scaler.shape)
st.write("Kích thước tập kiểm tra:", X_test_scaler.shape)

# Model
st.header("Xây dựng mô hình SVM")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaler, y_train)
# st.write("Mô hình SVM đã được huấn luyện thành công.")

# Đánh giá mô hình
st.subheader("Đánh giá mô hình với SVM")
y_pred = svm_model.predict(X_test_scaler)
st.write(f"Accuracy {accuracy_score(y_test,y_pred)}")
st.write(f"Precision {precision_score(y_test,y_pred)}")
st.write(f"Recall {recall_score(y_test,y_pred)}")
st.write(f"F1 Score {f1_score(y_test,y_pred)}")

st.subheader("Classification Report")
st.text(classification_report(y_test,y_pred,target_names=['Melanoma','Benign']))
# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
st.pyplot(fig)
plt.close(fig)

st.write("""
            Nhận xét:
            - Mô hình SVM đạt độ chính xác cao trên cả 4 chỉ số: Accuracy, Precision, Recall và F1 Score, cho thấy khả năng phân loại tốt giữa hai lớp Melanoma và Benign.
            - Confusion Matrix cho thấy số lượng dự đoán đúng và sai cho cả hai lớp, với số lượng dự đoán đúng (True Positives và True Negatives) chiếm phần lớn, minh chứng cho hiệu suất mạnh mẽ của mô hình.
            - Tuy nhiên, vẫn còn một số dự đoán sai (False Positives và False Negatives), điều này cho thấy có thể có những trường hợp mà mô hình gặp khó khăn trong việc phân biệt giữa hai lớp. Việc này có thể do sự tương đồng trong các đặc trưng của một số mẫu hoặc do giới hạn của mô hình SVM với các tham số hiện tại.
            - Để cải thiện hơn nữa, có thể xem xét việc tinh chỉnh tham số mô hình, sử dụng các kỹ thuật tiền xử lý dữ liệu nâng cao hơn, hoặc thử nghiệm với các thuật toán phân loại khác để so sánh hiệu suất.
            """)
# ROC Curve
st.subheader("ROC Curve") 
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc="lower right")
st.pyplot(fig)
plt.close(fig)

st.write("""
            Nhận xét:
            - Đường cong ROC cho thấy mô hình SVM có khả năng phân biệt tốt giữa hai lớp Melanoma và Benign, với AUC đạt giá trị cao (gần 1), minh chứng cho hiệu suất xuất sắc của mô hình.
            - Đường cong nằm gần góc trên bên trái của biểu đồ, cho thấy tỷ lệ True Positive Rate cao trong khi giữ tỷ lệ False Positive Rate thấp, điều này rất quan trọng trong các ứng dụng y tế nơi việc phát hiện chính xác các ca bệnh là ưu tiên hàng đầu.
            - Tuy nhiên, vẫn cần lưu ý rằng việc tối ưu hóa thêm có thể cần thiết để giảm thiểu các trường hợp False Positives và False Negatives, đặc biệt trong bối cảnh chẩn đoán ung thư da, nơi những sai sót này có thể dẫn đến hậu quả nghiêm trọng.
            - Để cải thiện hơn nữa, có thể xem xét việc tinh chỉnh tham số mô hình, sử dụng các kỹ thuật tiền xử lý dữ liệu nâng cao hơn, hoặc thử nghiệm với các thuật toán phân loại khác để so sánh hiệu suất.
            """)

st.subheader("Decision Boundary sau khi giảm chiều dữ liệu với PCA")
# Apply PCA to reduce the data to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
x_train_pca = pca.fit_transform(X_train_scaler)
x_test_pca = pca.transform(X_test_scaler)

# Train a new SVM model on the PCA-transformed data
svm_model_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model_pca.fit(x_train_pca, y_train)

# Now plot the decision boundary using the PCA-transformed data and the new model
plot_decision_boundary(x_test_pca, y_test, svm_model_pca)

# Dự đoán 

st.sidebar.subheader("Dự đoán với dữ liệu mới")
age = st.sidebar.number_input("Nhập tuổi bệnh nhân:", min_value=0, max_value=120, value=30)
sex = st.sidebar.selectbox("Chọn giới tính:", options=["female", "male"])
localization = st.sidebar.selectbox("Chọn vị trí tổn thương:", options=sorted(data["localization"].unique()))
# Chuẩn bị dữ liệu đầu vào

input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "male" else 0],
    "localization": data_no_null['localization'].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1).get(localization, -1)  
})
# Xử lý dữ liệu thiếu và chuẩn hóa
input_data_impute = impute.transform(input_data)
input_data_scaler = scaler.transform(input_data_impute)
# Dự đoán
if st.sidebar.button("Predict"):
    prediction = svm_model.predict(input_data_scaler)
    st.sidebar.write("Kết quả dự đoán:")
    if prediction[0] == 1:
        st.sidebar.success("Bệnh nhân có khả năng bị Ung thư da (Melanoma).")
    else:
        st.sidebar.warning("Bệnh nhân có khả năng không bị Ung thư da (Benign).")