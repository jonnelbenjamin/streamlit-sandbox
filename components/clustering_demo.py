# components/clustering_demo.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

def plot_clusters(data, labels, algorithm_name):
    """Visualize clusters using Plotly"""
    fig = px.scatter(
        x=data[:, 0], 
        y=data[:, 1], 
        color=labels,
        title=f'{algorithm_name} Clustering (n_clusters={len(np.unique(labels))})',
        labels={'x': 'Feature 1', 'y': 'Feature 2'},
        color_continuous_scale=px.colors.qualitative.Alphabet
    )
    fig.update_traces(marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

def plot_elbow_method(data, max_clusters=10):
    """Calculate and plot inertia for elbow method"""
    inertias = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_clusters+1), inertias, 'bo-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)
    
    return inertias

def plot_silhouette_analysis(data, max_clusters=10):
    """Perform silhouette analysis for different cluster counts"""
    silhouette_scores = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(2, max_clusters+1), silhouette_scores, 'go-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis')
    st.pyplot(fig)
    
    return silhouette_scores

def plot_k_distance(data, k=4):
    """Plot k-distance graph for DBSCAN eps parameter estimation"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    
    distances = np.sort(distances[:, k-1], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances)
    ax.set_xlabel('Points sorted by distance')
    ax.set_ylabel(f'{k}-NN distance')
    ax.set_title('K-Distance Graph for DBSCAN Eps Estimation')
    ax.axhline(y=st.session_state.dbscan_eps, color='r', linestyle='--')
    st.pyplot(fig)

def show_clustering_demo():
    """Demonstrate clustering algorithms with explanations"""
    
    st.title("Clustering Algorithms Demonstration")
    st.write("""
    This component showcases different clustering algorithms and their evaluation metrics.
    Explore how each algorithm performs on different types of data distributions.
    """)
    
    # Dataset selection
    st.sidebar.header("Dataset Configuration")
    dataset_type = st.sidebar.selectbox(
        "Select dataset type",
        ["Blobs", "Moons", "Anisotropic", "Noisy Circles"],
        index=0
    )
    
    n_samples = st.sidebar.slider("Number of samples", 100, 2000, 500)
    noise_level = st.sidebar.slider("Noise level", 0.01, 0.5, 0.05, disabled=dataset_type=="Blobs")
    
    # Generate dataset
    if dataset_type == "Blobs":
        data, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)
    elif dataset_type == "Moons":
        data, _ = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
    elif dataset_type == "Anisotropic":
        X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        data = np.dot(X, transformation)
    else:  # Noisy Circles
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Show raw data
    st.subheader("Raw Data Visualization")
    fig = px.scatter(x=data[:, 0], y=data[:, 1], title=f'{dataset_type} Dataset (n={n_samples})')
    st.plotly_chart(fig, use_container_width=True)
    
    # Algorithm selection
    st.sidebar.header("Algorithm Selection")
    algorithm = st.sidebar.selectbox(
        "Select clustering algorithm",
        ["K-Means", "DBSCAN", "Agglomerative"],
        index=0
    )
    
    # Algorithm parameters
    if algorithm == "K-Means":
        st.sidebar.subheader("K-Means Parameters")
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
        max_iter = st.sidebar.slider("Max iterations", 100, 500, 300)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Evaluation metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        
        # Results
        st.subheader("K-Means Clustering Results")
        plot_clusters(data, labels, "K-Means")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Elbow Method Analysis**")
            inertias = plot_elbow_method(data)
        
        with col2:
            st.markdown("**Silhouette Analysis**")
            silhouette_scores = plot_silhouette_analysis(data)
        
        st.subheader("Performance Metrics")
        st.write(f"""
        - **Inertia**: {inertia:.2f} (Sum of squared distances to nearest cluster center)
        - **Silhouette Score**: {silhouette:.2f} (Higher is better, range [-1, 1])
        - **Davies-Bouldin Index**: {davies_bouldin:.2f} (Lower is better)
        - **Calinski-Harabasz Index**: {calinski_harabasz:.2f} (Higher is better)
        """)
        
        st.markdown("""
        **Interpreting K-Means Metrics:**
        - **Inertia**: Measures how internally coherent clusters are. Look for the "elbow" point where inertia stops decreasing rapidly.
        - **Silhouette Score**: Measures how similar an object is to its own cluster vs other clusters. Values near +1 indicate good clustering.
        - **Davies-Bouldin**: Ratio of within-cluster distances to between-cluster distances. Lower values indicate better clustering.
        - **Calinski-Harabasz**: Ratio of between-clusters dispersion to within-cluster dispersion. Higher values indicate better clustering.
        """)
    
    elif algorithm == "DBSCAN":
        st.sidebar.subheader("DBSCAN Parameters")
        if 'dbscan_eps' not in st.session_state:
            st.session_state.dbscan_eps = 0.5
        
        eps = st.sidebar.slider(
            "Epsilon (eps)", 
            0.1, 1.0, 0.5, 0.01,
            key='dbscan_eps',
            help="Maximum distance between two samples for one to be considered in the neighborhood of the other"
        )
        min_samples = st.sidebar.slider(
            "Minimum samples", 
            1, 20, 5,
            help="Number of samples in a neighborhood for a point to be considered a core point"
        )
        
        # K-distance graph for eps estimation
        st.subheader("DBSCAN Parameter Guidance")
        st.write("""
        Use the k-distance graph below to help determine an appropriate eps value.
        Look for the 'knee' point where the distance starts increasing rapidly.
        """)
        plot_k_distance(data, k=min_samples)
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Count clusters (ignoring noise if present)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Evaluation metrics (only if we have clusters)
        if n_clusters > 0:
            silhouette = silhouette_score(data, labels) if n_clusters > 1 else None
            davies_bouldin = davies_bouldin_score(data, labels) if n_clusters > 1 else None
            calinski_harabasz = calinski_harabasz_score(data, labels) if n_clusters > 1 else None
        
        # Results
        st.subheader("DBSCAN Clustering Results")
        plot_clusters(data, labels, "DBSCAN")
        
        st.write(f"Number of clusters found: {n_clusters}")
        st.write(f"Number of noise points: {np.sum(labels == -1)}")
        
        if n_clusters > 0:
            st.subheader("Performance Metrics")
            st.write(f"""
            - **Silhouette Score**: {silhouette:.2f if silhouette else 'N/A (only 1 cluster)'}
            - **Davies-Bouldin Index**: {davies_bouldin:.2f if davies_bouldin else 'N/A (only 1 cluster)'}
            - **Calinski-Harabasz Index**: {calinski_harabasz:.2f if calinski_harabasz else 'N/A (only 1 cluster)'}
            """)
        
        st.markdown("""
        **Interpreting DBSCAN Results:**
        - Points labeled -1 are considered noise/outliers
        - DBSCAN doesn't require specifying number of clusters upfront
        - Works well with arbitrary-shaped clusters and noisy data
        - Performance depends heavily on proper eps and min_samples selection
        """)
    
    elif algorithm == "Agglomerative":
        st.sidebar.subheader("Agglomerative Parameters")
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
        linkage = st.sidebar.selectbox(
            "Linkage method",
            ["ward", "complete", "average", "single"],
            index=0,
            help="Which linkage criterion to use. Ward minimizes variance, others use max/avg/min distances"
        )
        
        # Run Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(data)
        
        # Evaluation metrics
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        
        # Results
        st.subheader("Agglomerative Clustering Results")
        plot_clusters(data, labels, "Agglomerative")
        
        st.subheader("Performance Metrics")
        st.write(f"""
        - **Silhouette Score**: {silhouette:.2f}
        - **Davies-Bouldin Index**: {davies_bouldin:.2f}
        - **Calinski-Harabasz Index**: {calinski_harabasz:.2f}
        """)
        
        st.markdown("""
        **Interpreting Agglomerative Clustering:**
        - Builds clusters hierarchically (bottom-up approach)
        - Different linkage methods produce different cluster shapes:
          - **Ward**: Minimizes variance (creates spherical clusters)
          - **Complete**: Uses maximum distances (can handle non-elliptical shapes)
          - **Average**: Uses average distances (compromise between ward and complete)
          - **Single**: Uses minimum distances (can create long chains)
        - Produces a dendrogram that can be useful for understanding data structure
        """)

if __name__ == "__main__":
    show_clustering_demo()