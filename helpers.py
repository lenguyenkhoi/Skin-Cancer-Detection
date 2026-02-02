import math
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
# EDA
def plot_all_histograms(data, title_prefix=""):
    num_cols = data.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    
# Correlation matrix 
def plot_correlation_matrix(data, method="pearson"):
    corr = data.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr,annot=True, fmt=".2f",cmap="coolwarm",square=True,linewidths=0.5,cbar=True,ax=ax)

    ax.set_title(f"Correlation Matrix ({method.capitalize()})")
    st.pyplot(fig)
    plt.close(fig)
    
    
# Imbalance
def plot_class_distribution(data):
    fraud_counts = data['diagnosis_binary'].value_counts().sort_index()

    fig = plt.figure(figsize=(10, 5))

    # Bar plot
    plt.subplot(1, 2, 1)
    colors = ['#2ecc71', '#e74c3c']
    plt.bar(
        ['Melanoma (0)', 'Benign (1)'],
        fraud_counts.values,
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Imbalanced Dataset - Bar Chart')

    for i, v in enumerate(fraud_counts.values):
        plt.text(i, v + max(fraud_counts.values)*0.02, str(v),ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Imbalance multi_class
def plot_multiclass_distribution(data, labels_col):
    class_counts = data[labels_col].value_counts().sort_index()

    fig = plt.figure(figsize=(10, 5))

    # random list column
    colors = plt.cm.tab10(range(len(class_counts)))

    plt.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=colors,
        alpha=0.8,
        edgecolor='black'
    )

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Multi-class Distribution')

    # Hiển thị số lượng trên cột
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + max(class_counts.values)*0.02,
                 str(v), ha='center', fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# Plot decision boundary 
def plot_decision_boundary(X, y, model):
    # Tạo meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    # Dự đoán trên toàn vùng
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Vẽ
    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(
        xx, yy, Z,
        cmap=plt.cm.coolwarm,
        alpha=0.8
    )

    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        edgecolors='k',
        s=50
    )

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('SVM Decision Boundary (PCA 2D)')

    cbar = fig.colorbar(contour)
    cbar.set_label('Class (0 / 1)')

    st.pyplot(fig)
    plt.close(fig)
    
def downsample_multiclass(df, target_col):
    
    # Tìm số lượng nhỏ nhất
    min_count = df[target_col].value_counts().min()

    downsampled_list = []

    for cls in df[target_col].unique():
        
        df_class = df[df[target_col] == cls]

        df_downsampled = resample(
            df_class,
            replace=False,
            n_samples=min_count,
            random_state=42
        )

        downsampled_list.append(df_downsampled)

    df_balanced = pd.concat(downsampled_list)

    return df_balanced.sample(frac=1, random_state=42)


# 
def plot_multiclass_roc_auc_streamlit(model, X_test, y_test, class_names=None):
    y_score = model.predict_proba(X_test)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    # Tính ROC
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Micro Average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(),y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    #Macro Average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    
    fig, ax = plt.subplots(figsize=(8,6))
    # ROC từng class
    for i in range(n_classes):
        if class_names is not None:
            label_name = class_names[i]
        else:
            label_name = f"Class {classes[i]}"
        ax.plot(fpr[i], tpr[i], label=f"{label_name} (AUC = {roc_auc[i]:.2f})")

    # Micro & Macro
    ax.plot(fpr_micro,tpr_micro,linestyle=":",label=f"Micro Avg (AUC = {roc_auc_micro:.2f})")

    ax.plot(all_fpr,mean_tpr,linestyle=":",label=f"Macro Avg (AUC = {roc_auc_macro:.2f})")

    ax.plot([0,1], [0,1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curve")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)
