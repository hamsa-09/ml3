import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('patient_data.csv')

# Select relevant features
features = data[['Age', 'Blood Pressure', 'Cholesterol', 'Blood Sugar', 'Height', 'Weight', 'BMI', 
                 'Pulse Rate', 'Respiratory Rate', 'Oxygen Saturation', 'Creatinine Level', 'Hemoglobin Level']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Function for K-Means Clustering
def kmeans_clustering(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    kmeans_labels = kmeans.labels_
    return kmeans_labels

# Function for EM Clustering
def em_clustering(n_components):
    em = GaussianMixture(n_components=n_components, random_state=42)
    em.fit(scaled_features)
    em_labels = em.predict(scaled_features)
    return em_labels

# Function for evaluating clustering performance
def evaluate_clustering(labels):
    silhouette = silhouette_score(scaled_features, labels)
    calinski = calinski_harabasz_score(scaled_features, labels)
    return silhouette, calinski

# Function to visualize clusters using PCA
def visualize_clusters(labels):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='coolwarm', s=50, edgecolor='black')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cluster Visualization')
    return plt

# Streamlit Title with custom styles
st.markdown("<h1 style='text-align: center; color: #D32F2F;'>🧬 Patient Clustering for Personalized Treatment 🧬</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Explore the power of clustering for optimized patient treatment plans</h4>", unsafe_allow_html=True)

# Sidebar for Cluster Settings
st.sidebar.markdown("<h2 style='color: #D32F2F;'>🔧 Cluster Settings</h2>", unsafe_allow_html=True)
n_clusters = st.sidebar.slider("Select Number of Clusters (K-Means)", 2, 10, 5, help="Set the number of clusters for K-Means")
n_components = st.sidebar.slider("Select Number of Components (EM)", 2, 10, 5, help="Set the number of components for EM")

# Perform clustering
kmeans_labels = kmeans_clustering(n_clusters)
em_labels = em_clustering(n_components)

# Evaluate the clustering results
kmeans_silhouette, kmeans_calinski = evaluate_clustering(kmeans_labels)
em_silhouette, em_calinski = evaluate_clustering(em_labels)

# Add some space
st.markdown("<hr>", unsafe_allow_html=True)

# Display results
st.markdown("<h2 style='text-align: center; color: #303F9F;'>🔍 Clustering Performance Metrics</h2>", unsafe_allow_html=True)

# Add metrics to display on the app
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⚙️ K-Means Clustering")
    st.metric("Silhouette Score", f"{kmeans_silhouette:.3f}")
    st.metric("Calinski-Harabasz Index", f"{kmeans_calinski:.3f}")

with col2:
    st.markdown("#### 🔄 Expectation-Maximization (EM) Clustering")
    st.metric("Silhouette Score", f"{em_silhouette:.3f}")
    st.metric("Calinski-Harabasz Index", f"{em_calinski:.3f}")

# Add some space
st.markdown("<hr>", unsafe_allow_html=True)

# Visualize the clusters using PCA
st.markdown("<h2 style='text-align: center; color: #388E3C;'>🌐 Cluster Visualization</h2>", unsafe_allow_html=True)
st.markdown("#### 🧩 **K-Means Clustering** Visualization")

plt.figure(figsize=(7, 5))
plt = visualize_clusters(kmeans_labels)
st.pyplot(plt)

st.markdown("#### 🧩 **Expectation-Maximization (EM)** Clustering Visualization")

plt.figure(figsize=(7, 5))
plt = visualize_clusters(em_labels)
st.pyplot(plt)

# Conclusion section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #F57C00;'>📊 Conclusion</h2>", unsafe_allow_html=True)

if kmeans_silhouette > em_silhouette and kmeans_calinski > em_calinski:
    st.success("K-Means performs better based on both metrics (Silhouette Score and Calinski-Harabasz Index).")
elif em_silhouette > kmeans_silhouette and em_calinski > kmeans_calinski:
    st.success("EM performs better based on both metrics (Silhouette Score and Calinski-Harabasz Index).")
else:
    st.warning("Mixed results: K-Means and EM perform differently based on various metrics.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: #455A64;'>Data-driven insights for personalized healthcare planning © 2024</h6>", unsafe_allow_html=True)
