import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Indian Startups Clustering Analysis", layout="wide")

# Custom color palette
COLOR_PALETTE = px.colors.qualitative.Bold

# Custom CSS to style the app
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #43a047;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1E88E5;
        border-radius: 4px;
    }
    .stSelectbox [data-baseweb="select"] {
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    return None

# Data preprocessing function
@st.cache_data
# Preprocessing function
# Preprocessing function
def preprocess_data(df, features):
    try:
        X = df[features].copy()
        X = X.replace('not available', np.nan)
        
        for col in X.columns:
            if col == 'No. of Employees':
                X[col] = X[col].apply(convert_employees)
            elif col == 'Funding Amount in $':
                X[col] = X[col].apply(convert_funding)
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_imputed)
        
        return X_normalized, X
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None

def convert_employees(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = '-'.join(sorted(set(value.split('-'))))
        value = value.replace(',', '').strip().lower()
        if '-' in value:
            low, high = value.split('-')
            try:
                return (float(low) + float(high)) / 2
            except ValueError:
                return np.nan
        elif '+' in value:
            try:
                return float(value.replace('+', ''))
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return np.nan

def convert_funding(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value.replace(',', '').strip())
        except ValueError:
            return np.nan
    return np.nan

# Clustering function
def perform_clustering(X, algorithm, n_clusters=None):
    try:
        if algorithm == 'K-Means':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == 'DBSCAN':
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'Gaussian Mixture':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        labels = model.fit_predict(X)
        
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
        else:
            silhouette = 0
            db_score = 0
        
        return labels, silhouette, db_score, model
    except Exception as e:
        st.error(f"Error performing clustering: {str(e)}")
        return None, None, None, None

# Plotting function
def plot_clusters(X, labels, feature_names, df):
    try:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        df_plot['Cluster'] = labels
        df_plot['Company'] = df['Company']
        
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            color='Cluster', hover_data=['Company'],
                            title="3D Cluster Visualization (PCA)")
        return fig
    except Exception as e:
        st.error(f"Error plotting clusters: {str(e)}")
        return None


# Function to generate insights
def generate_insights(df, labels):
    insights = []
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    # Preprocess 'No. of Employees' column
    def preprocess_employees(value):
        if isinstance(value, str):
            if '-' in value:
                low, high = map(lambda x: float(x.replace(',', '').strip()), value.split('-'))
                return (low + high) / 2
            elif '+' in value:
                return float(value.replace('+', '').replace(',', '').strip())
        return pd.to_numeric(value, errors='coerce')

    df_copy['No. of Employees'] = df_copy['No. of Employees'].apply(preprocess_employees)
    df_copy['Funding Amount in $'] = df_copy['Funding Amount in $'].apply(lambda x: pd.to_numeric(str(x).replace(',', ''), errors='coerce'))

    # Funding Tiers
    funding_mean = df_copy.groupby('Cluster')['Funding Amount in $'].mean().sort_values(ascending=False)
    top_cluster = funding_mean.index[0]
    bottom_cluster = funding_mean.index[-1]
    insights.append(f"Funding Tiers: Cluster {top_cluster} has the highest average funding (${funding_mean[top_cluster]:,.2f}), while Cluster {bottom_cluster} has the lowest (${funding_mean[bottom_cluster]:,.2f}).")
    
    # Growth Correlation
    employee_funding_corr = df_copy['No. of Employees'].corr(df_copy['Funding Amount in $'])
    insights.append(f"Growth Correlation: There's a {abs(employee_funding_corr):.2f} {'positive' if employee_funding_corr > 0 else 'negative'} correlation between employee count and funding received.")
    
    # Generational Trends
    year_mean = df_copy.groupby('Cluster')['Starting Year'].mean()
    oldest_cluster = year_mean.idxmin()
    newest_cluster = year_mean.idxmax()
    insights.append(f"Generational Trends: Cluster {oldest_cluster} represents the oldest startups (avg. year {year_mean[oldest_cluster]:.0f}), while Cluster {newest_cluster} represents the newest (avg. year {year_mean[newest_cluster]:.0f}).")
    
    # Industry Hotspots
    industry_counts = df_copy.groupby('Cluster')['Industries'].apply(lambda x: ', '.join(x).split(', ')).apply(pd.Series).stack().value_counts()
    top_industry = industry_counts.index[0]
    insights.append(f"Industry Hotspots: The most common industry across all clusters is {top_industry}, appearing {industry_counts[top_industry]} times.")
    
    # Geographical Patterns
    city_counts = df_copy.groupby('Cluster')['City'].value_counts()
    top_city = city_counts.index[0]
    insights.append(f"Geographical Patterns: The city with the highest representation in a single cluster is {top_city[1]} in Cluster {top_city[0]}, with {city_counts[top_city]} startups.")
    
    return insights

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Clustering Analysis", "Comparison", "Insights"])

# Data upload in sidebar
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Load data
df = load_data(uploaded_file)

if df is None:
    st.warning("Please upload a CSV file to begin the analysis.")
else:
    # Main content
    if page == "Home":
        st.title("🚀 Clustering Analysis of Top 300 Indian Startups")
        st.write("Welcome to our premium startup analysis tool. Dive into the world of Indian startups and uncover hidden patterns!")
        
        # Dataset overview
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Startups", f"{len(df):,}")
        with col2:
            st.metric("Average Funding", f"${df['Funding Amount in $'].mean():,.2f}")
        with col3:
            st.metric("Most Common City", df['City'].mode()[0])
        
        # Most common industries
        st.subheader("🏭 Top 5 Industries")
        top_industries = df['Industries'].str.split(', ', expand=True).stack().value_counts().head(5)
        fig = px.pie(values=top_industries.values, names=top_industries.index, title="Top 5 Industries",
                     color_discrete_sequence=COLOR_PALETTE)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Data Exploration":
        st.title("🔍 Data Exploration")
        
        # Interactive table
        st.subheader("Dataset Viewer")
        st.dataframe(df.style.background_gradient(cmap='Blues'))
        
        # Distribution of startups by city
        st.subheader("🏙️ Distribution of Startups by City")
        city_counts = df['City'].value_counts().head(10)
        fig = px.bar(x=city_counts.index, y=city_counts.values, labels={'x': 'City', 'y': 'Number of Startups'},
                     color=city_counts.values, color_continuous_scale=COLOR_PALETTE)
        fig.update_layout(title="Top 10 Cities by Number of Startups")
        st.plotly_chart(fig, use_container_width=True)
        
        # Funding Amount Distribution
        st.subheader("💰 Funding Amount Distribution")
        fig = px.histogram(df, x="Funding Amount in $", nbins=50, title="Distribution of Funding Amounts",
                           color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=COLOR_PALETTE)
        fig.update_layout(title="Correlation Heatmap of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Clustering Analysis":
        st.title("🧮 Clustering Analysis")
        
        features = st.multiselect("Select features for clustering", 
                                ['Starting Year', 'No. of Employees', 'Funding Amount in $', 'Funding Round', 'No. of Investors'],
                                default=['Funding Amount in $', 'No. of Employees', 'Starting Year'])
        
        # Data quality check
        st.subheader("Data Quality Check")
        for feature in features:
            missing = df[feature].isna().sum()
            not_available = (df[feature] == 'not available').sum()
            st.write(f"{feature}:")
            st.write(f"  - Missing values: {missing}")
            st.write(f"  - 'Not available' values: {not_available}")
        
        algorithm = st.selectbox("Select clustering algorithm", ['K-Means', 'DBSCAN', 'Agglomerative', 'Gaussian Mixture'])
        
        if algorithm in ['K-Means', 'Agglomerative', 'Gaussian Mixture']:
            n_clusters = st.slider("Number of clusters", 2, 10, 5)
        else:
            n_clusters = None
        
        if st.button("Perform Clustering"):
            with st.spinner("Preprocessing data..."):
                X_normalized, X_preprocessed = preprocess_data(df, features)
            
            if X_normalized is not None and X_preprocessed is not None:
                st.success("Preprocessing completed successfully.")
                
                with st.spinner("Performing clustering..."):
                    labels, silhouette, db_score, model = perform_clustering(X_normalized, algorithm, n_clusters)
                
                if labels is not None:
                    st.session_state['labels'] = labels
                    st.subheader("Clustering Results")
                    fig = plot_clusters(X_normalized, labels, features, df)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette:.2f}")
                    with col2:
                        st.metric("Davies-Bouldin Score", f"{db_score:.2f}")
                    
                    if algorithm in ['K-Means', 'Gaussian Mixture']:
                        if algorithm == 'K-Means':
                            centers = model.cluster_centers_
                            st.subheader("Cluster Centers")
                        else:  # Gaussian Mixture
                            centers = model.means_
                            st.subheader("Gaussian Component Means")
                        
                        centers_df = pd.DataFrame(centers, columns=features)
                        st.dataframe(centers_df.style.background_gradient(cmap='YlOrRd'))
                    
                    st.subheader("Cluster Characteristics")
                    X_preprocessed['Cluster'] = labels
                    for cluster in range(len(set(labels))):
                        st.write(f"Cluster {cluster}")
                        cluster_data = X_preprocessed[X_preprocessed['Cluster'] == cluster]
                        
                        cols = st.columns(len(features))
                        for i, feature in enumerate(features):
                            with cols[i]:
                                avg_value = cluster_data[feature].mean()
                                if feature == 'Funding Amount in $':
                                    st.metric(f"Avg {feature}", f"${avg_value:,.2f}" if pd.notnull(avg_value) else "N/A")
                                elif feature == 'No. of Employees':
                                    st.metric(f"Avg {feature}", f"{avg_value:,.0f}" if pd.notnull(avg_value) else "N/A")
                                else:
                                    st.metric(f"Avg {feature}", f"{avg_value:.2f}" if pd.notnull(avg_value) else "N/A")
                else:
                    st.error("Clustering failed. Please try different parameters or check your data.")
            else:
                st.error("Preprocessing failed. Please check your data and try again.")

    elif page == "Comparison":
        st.title("🔬 Algorithm Comparison")
        
        features = st.multiselect("Select features for clustering", 
                                ['Starting Year', 'No. of Employees', 'Funding Amount in $', 'Funding Round', 'No. of Investors'],
                                default=['Funding Amount in $', 'No. of Employees', 'Starting Year'])
        
        # Data quality check
        st.subheader("Data Quality Check")
        for feature in features:
            missing = df[feature].isna().sum()
            not_available = (df[feature] == 'not available').sum()
            st.write(f"{feature}:")
            st.write(f"  - Missing values: {missing}")
            st.write(f"  - 'Not available' values: {not_available}")
        
        if st.button("Compare Algorithms"):
            with st.spinner("Preprocessing data..."):
                X_normalized, X_preprocessed = preprocess_data(df, features)
            
            if X_normalized is not None and X_preprocessed is not None:
                st.success("Preprocessing completed successfully.")
                
                # Apply PCA if more than 2 features are selected
                if X_normalized.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_normalized = pca.fit_transform(X_normalized)
                    st.info(f"PCA applied to reduce dimensions to 2. Explained variance ratio: {pca.explained_variance_ratio_}")
                
                with st.spinner("Comparing algorithms..."):
                    results = []
                    algorithms = ['K-Means', 'DBSCAN', 'Agglomerative', 'Gaussian Mixture']
                    for algorithm in algorithms:
                        if algorithm in ['K-Means', 'Agglomerative', 'Gaussian Mixture']:
                            for n_clusters in range(2, 11):
                                try:
                                    labels, silhouette, db_score, _ = perform_clustering(X_normalized, algorithm, n_clusters)
                                    if labels is not None:
                                        results.append({'Algorithm': f'{algorithm} (k={n_clusters})', 'Silhouette Score': silhouette, 'Davies-Bouldin Score': db_score})
                                except Exception as e:
                                    st.warning(f"Error with {algorithm} (k={n_clusters}): {str(e)}")
                        else:
                            try:
                                labels, silhouette, db_score, _ = perform_clustering(X_normalized, algorithm)
                                if labels is not None:
                                    results.append({'Algorithm': algorithm, 'Silhouette Score': silhouette, 'Davies-Bouldin Score': db_score})
                            except Exception as e:
                                st.warning(f"Error with {algorithm}: {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    st.subheader("Algorithm Performance Comparison")
                    fig = px.scatter(results_df, x='Silhouette Score', y='Davies-Bouldin Score', 
                                    color='Algorithm', hover_data=['Algorithm'],
                                    title="Clustering Algorithm Performance")
                    fig.update_traces(marker=dict(size=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Best Performing Algorithms")
                    best_silhouette = results_df.loc[results_df['Silhouette Score'].idxmax()]
                    best_db = results_df.loc[results_df['Davies-Bouldin Score'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Silhouette Score", f"{best_silhouette['Silhouette Score']:.2f}")
                        st.write(f"Algorithm: {best_silhouette['Algorithm']}")
                    with col2:
                        st.metric("Best Davies-Bouldin Score", f"{best_db['Davies-Bouldin Score']:.2f}")
                        st.write(f"Algorithm: {best_db['Algorithm']}")
                    
                    st.subheader("Detailed Results")
                    st.dataframe(results_df.style.background_gradient(cmap='RdYlGn', subset=['Silhouette Score'])
                                .background_gradient(cmap='RdYlGn_r', subset=['Davies-Bouldin Score']))
                    
                    st.subheader("Algorithm Descriptions")
                    st.write("""
                    - **K-Means**: Partitions the data into k clusters, each represented by its centroid.
                    - **DBSCAN**: Density-based clustering that groups together points that are closely packed together.
                    - **Agglomerative**: Hierarchical clustering that builds nested clusters by merging them successively.
                    - **Gaussian Mixture**: Probabilistic model that assumes the data is generated from a mixture of a finite number of Gaussian distributions.
                    """)
                else:
                    st.warning("No valid clustering results were obtained. Please try different parameters or check your data.")
            else:
                st.error("Preprocessing failed. Please check your data and try again.")
    elif page == "Insights":
        st.title("💡 Insights")
        
        if 'labels' not in st.session_state:
            st.warning("Please perform clustering analysis first to generate insights.")
        else:
            try:
                with st.spinner("Generating insights..."):
                    insights = generate_insights(df, st.session_state['labels'])
                
                for insight in insights:
                    st.write("• " + insight)
                
                st.subheader("Strategic Implications")
                st.write("""
                These insights can be leveraged by:
                - 🎯 Investors: To identify promising startups and emerging trends for potential investments.
                - 💼 Entrepreneurs: To understand the current startup landscape and position their ventures strategically.
                - 🏛️ Policymakers: To develop targeted policies that foster startup ecosystems in various regions.
                """)
                
                # Additional visualizations based on insights
                st.subheader("Funding Distribution Across Clusters")
                df_temp = df.copy()
                df_temp['Cluster'] = st.session_state['labels']
                df_temp['Funding Amount in $'] = df_temp['Funding Amount in $'].apply(lambda x: pd.to_numeric(str(x).replace(',', ''), errors='coerce'))
                fig = px.box(df_temp, x='Cluster', y='Funding Amount in $', color='Cluster',
                            title="Funding Distribution by Cluster")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Average Employees per Cluster")
                df_temp['No. of Employees'] = df_temp['No. of Employees'].apply(lambda x: pd.to_numeric(x.split('-')[0] if isinstance(x, str) and '-' in x else x, errors='coerce'))
                avg_employees = df_temp.groupby('Cluster')['No. of Employees'].mean().sort_values(ascending=False)
                fig = px.bar(x=avg_employees.index, y=avg_employees.values, 
                            labels={'x': 'Cluster', 'y': 'Average Number of Employees'},
                            title="Average Number of Employees by Cluster")
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"An error occurred while generating insights: {str(e)}")
                st.write("Please check your data and try again.")