import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px

def kmeans_clustering():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, iris.target_names

def plot_2d_scatter(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['sepal length (cm)'], 
        y=df['sepal width (cm)'], hue=df['Cluster'], palette='viridis')
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("K-Means Clustering (2D View)")
    plt.savefig("static/plot_2d.png")
    plt.close()

def plot_3d_scatter(df):
    fig = px.scatter_3d(df, x='sepal length (cm)', y='sepal width (cm)', 
        z='petal length (cm)', color=df['Cluster'].astype(str), 
        title="K-Means Clustering (3D View)")
    fig.write_html("static/plot_3d.html")

