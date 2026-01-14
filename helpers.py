import math
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import streamlit as st

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