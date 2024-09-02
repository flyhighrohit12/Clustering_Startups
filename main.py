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
    
def preprocess_employees(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = value.replace(',', '').strip().lower()
        if '-' in value:
            low, high = map(lambda x: float(x.strip()), value.split('-'))
            return (low + high) / 2
        elif '+' in value:
            return float(value.replace('+', ''))
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return np.nan

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
        st.title("🚀 Indian Startup Ecosystem Analysis")
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Uncover insights from the top 300 Indian startups</p>', unsafe_allow_html=True)

        # Key Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_funding = df['Funding Amount in $'].sum()
        avg_funding = df['Funding Amount in $'].mean()
        total_startups = len(df)
        unique_cities = df['City'].nunique()

        col1.metric("Total Funding", f"${total_funding:,.0f}")
        col2.metric("Average Funding", f"${avg_funding:,.0f}")
        col3.metric("Total Startups", f"{total_startups}")
        col4.metric("Startup Hubs", f"{unique_cities}")

        # Funding Over Time
        st.subheader("📈 Funding Trends Over the Years")
        yearly_funding = df.groupby('Starting Year')['Funding Amount in $'].sum().reset_index()
        fig = px.area(yearly_funding, x='Starting Year', y='Funding Amount in $',
                    title="Total Funding by Year",
                    labels={'Funding Amount in $': 'Total Funding ($)', 'Starting Year': 'Year'})
        fig.update_layout(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        # Top Industries and Cities
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏭 Top Industries")
            industries = df['Industries'].str.split(', ', expand=True).stack().value_counts().head(5)
            fig = px.pie(values=industries.values, names=industries.index, hole=0.3,
                        title="Top 5 Industries")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏙️ Top Startup Hubs")
            cities = df['City'].value_counts().head(5)
            fig = px.bar(x=cities.index, y=cities.values,
                        labels={'x': 'City', 'y': 'Number of Startups'},
                        title="Top 5 Cities by Number of Startups")
            st.plotly_chart(fig, use_container_width=True)

        # Funding Distribution
        st.subheader("💰 Funding Distribution")
        fig = px.box(df, y="Funding Amount in $", points="all",
                    title="Distribution of Funding Amounts")
        st.plotly_chart(fig, use_container_width=True)

        # Recent Success Stories
        st.subheader("🌟 Recent Success Stories")
        recent_successes = df.nlargest(5, 'Funding Amount in $')[['Company', 'City', 'Industries', 'Funding Amount in $']]
        for _, company in recent_successes.iterrows():
            with st.expander(f"{company['Company']} - ${company['Funding Amount in $']:,.0f}"):
                st.write(f"**Location:** {company['City']}")
                st.write(f"**Industry:** {company['Industries']}")
                st.write(f"**Funding:** ${company['Funding Amount in $']:,.0f}")

        # Call to Action
        st.markdown("---")
        st.markdown('<p class="big-font">Ready to dive deeper?</p>', unsafe_allow_html=True)
        st.write("Explore our Data Exploration and Clustering Analysis tabs to uncover more insights about the Indian startup ecosystem.")

        # Footer
        st.markdown("---")
        st.markdown("*Data last updated: [Insert Date]*")
        st.markdown("Powered by Streamlit and Plotly | Created by [Your Name/Company]")

    elif page == "Data Exploration":
        st.title("🔍 Data Exploration")
        
        # Quick Data Quality Check
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning(f"There are {missing_data.sum()} missing values in the dataset. Consider handling these before analysis.")
        else:
            st.success("No missing values found in the dataset.")

        # Funding Distribution
        st.subheader("💰 Funding Distribution")
        df['Funding Amount in $'] = pd.to_numeric(df['Funding Amount in $'].replace('[\$,]', '', regex=True), errors='coerce')
        
        fig = px.histogram(df, x="Funding Amount in $", nbins=50, 
                        title="Distribution of Funding Amounts",
                        labels={"Funding Amount in $": "Funding Amount (USD)"})
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        median_funding = df['Funding Amount in $'].median()
        mean_funding = df['Funding Amount in $'].mean()
        st.write(f"**Insight:** The median funding is ${median_funding:,.0f}, while the mean is ${mean_funding:,.0f}. "
                f"This suggests a right-skewed distribution, with a few startups receiving significantly high funding.")

        # Top Funded Startups
        st.subheader("🏆 Top 10 Funded Startups")
        top_funded = df.nlargest(10, 'Funding Amount in $')[['Company', 'Funding Amount in $', 'City']]
        fig = px.bar(top_funded, x='Company', y='Funding Amount in $', color='City',
                    title="Top 10 Funded Startups")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        top_company = top_funded.iloc[0]
        st.write(f"**Insight:** {top_company['Company']} from {top_company['City']} leads with ${top_company['Funding Amount in $']:,.0f} in funding. "
                f"The top 10 funded startups are from {top_funded['City'].nunique()} different cities.")

        # Startup Distribution by City
        st.subheader("🏙️ Startup Distribution by City")
        city_counts = df['City'].value_counts().head(10)
        fig = px.pie(values=city_counts.values, names=city_counts.index, title="Top 10 Cities by Number of Startups")
        st.plotly_chart(fig, use_container_width=True)

        top_city = city_counts.index[0]
        st.write(f"**Insight:** {top_city} leads with {city_counts[top_city]} startups, representing {city_counts[top_city]/len(df):.1%} of all startups in the dataset.")

        # Funding by Starting Year
        st.subheader("📅 Funding Trends Over Years")
        df['Starting Year'] = pd.to_numeric(df['Starting Year'], errors='coerce')
        yearly_funding = df.groupby('Starting Year')['Funding Amount in $'].mean().reset_index()
        fig = px.line(yearly_funding, x='Starting Year', y='Funding Amount in $',
                    title="Average Funding Amount by Starting Year")
        st.plotly_chart(fig, use_container_width=True)

        peak_year = yearly_funding.loc[yearly_funding['Funding Amount in $'].idxmax()]
        st.write(f"**Insight:** The peak average funding of ${peak_year['Funding Amount in $']:,.0f} was observed for startups founded in {peak_year['Starting Year']:.0f}.")

        # Industry Analysis
        st.subheader("🏭 Top Industries")
        industries = df['Industries'].str.split(', ', expand=True).stack().value_counts().head(10)
        fig = px.bar(x=industries.index, y=industries.values, title="Top 10 Industries")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        top_industry = industries.index[0]
        st.write(f"**Insight:** The most prevalent industry is {top_industry} with {industries[top_industry]} startups, "
                f"followed by {industries.index[1]} and {industries.index[2]}.")

        # Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap")
        numeric_cols = ['Starting Year', 'Funding Amount in $', 'Funding Round', 'No. of Investors']
        df_corr = df[numeric_cols].copy()
        df_corr['No. of Employees'] = df['No. of Employees'].apply(preprocess_employees)
        df_corr['Funding Amount in $'] = pd.to_numeric(df_corr['Funding Amount in $'].replace('[\$,]', '', regex=True), errors='coerce')
        df_corr['Starting Year'] = pd.to_numeric(df_corr['Starting Year'], errors='coerce')
        
        corr = df_corr.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                        title="Correlation Heatmap of Key Features")
        st.plotly_chart(fig, use_container_width=True)

        max_corr = corr.max().sort_values(ascending=False).iloc[1]
        max_corr_features = corr.max().sort_values(ascending=False).index[:2]
        st.write(f"**Insight:** The strongest correlation ({max_corr:.2f}) is between {max_corr_features[0]} and {max_corr_features[1]}.")


    # Clustering Analysis Tab
    elif page == "Clustering Analysis":
        st.title("🧮 Clustering Analysis")
        
        features = st.multiselect("Select features for clustering", 
                                ['Starting Year', 'No. of Employees', 'Funding Amount in $', 'Funding Round', 'No. of Investors'],
                                default=['Funding Amount in $', 'No. of Employees', 'Starting Year'])
        
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
                    
                    st.write(f"**Insight:** The clustering algorithm identified {len(set(labels))} distinct groups. "
                            f"A Silhouette Score of {silhouette:.2f} suggests {'good' if silhouette > 0.5 else 'moderate' if silhouette > 0.3 else 'poor'} cluster separation. "
                            f"The Davies-Bouldin Score of {db_score:.2f} indicates {'low' if db_score < 0.5 else 'moderate' if db_score < 1 else 'high'} cluster overlap.")
                    
                    if algorithm in ['K-Means', 'Gaussian Mixture']:
                        if algorithm == 'K-Means':
                            centers = model.cluster_centers_
                            st.subheader("Cluster Centers")
                        else:  # Gaussian Mixture
                            centers = model.means_
                            st.subheader("Gaussian Component Means")
                        
                        centers_df = pd.DataFrame(centers, columns=features)
                        st.dataframe(centers_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                        
                        st.write(f"**Insight:** The cluster centers reveal distinct patterns among the startups. "
                                f"For example, Cluster 1 is characterized by {centers_df.iloc[0].idxmax()} "
                                f"with a value of {centers_df.iloc[0].max():.2f}, while Cluster 2 stands out in {centers_df.iloc[1].idxmax()} "
                                f"with {centers_df.iloc[1].max():.2f}.")
                    
                    st.subheader("Cluster Characteristics")
                    X_preprocessed['Cluster'] = labels
                    for cluster in range(len(set(labels))):
                        with st.expander(f"Cluster {cluster}"):
                            cluster_data = X_preprocessed[X_preprocessed['Cluster'] == cluster]
                            
                            cols = st.columns(len(features))
                            for i, feature in enumerate(features):
                                with cols[i]:
                                    avg_value = cluster_data[feature].mean()
                                    if feature == 'Funding Amount in $':
                                        st.metric(f"Avg {feature}", f"${avg_value:,.2f}")
                                    elif feature == 'No. of Employees':
                                        st.metric(f"Avg {feature}", f"{avg_value:,.0f}")
                                    else:
                                        st.metric(f"Avg {feature}", f"{avg_value:.2f}")
                            
                            st.write(f"**Insight:** Cluster {cluster} contains {len(cluster_data)} startups. "
                                    f"It's distinguished by {'high' if avg_value > X_preprocessed[feature].mean() else 'low'} {feature} "
                                    f"with an average of {avg_value:,.2f}.")
                else:
                    st.error("Clustering failed. Please try different parameters or check your data.")
            else:
                st.error("Preprocessing failed. Please check your data and try again.")

    # Comparison Tab
    elif page == "Comparison":
        st.title("🔬 Algorithm Comparison")
        
        features = st.multiselect("Select features for clustering", 
                                ['Starting Year', 'No. of Employees', 'Funding Amount in $', 'Funding Round', 'No. of Investors'],
                                default=['Funding Amount in $', 'No. of Employees', 'Starting Year'])
        
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
                    
                    best_silhouette = results_df.loc[results_df['Silhouette Score'].idxmax()]
                    best_db = results_df.loc[results_df['Davies-Bouldin Score'].idxmin()]
                    
                    st.write(f"**Insight:** The best performing algorithm based on Silhouette Score is {best_silhouette['Algorithm']} "
                            f"with a score of {best_silhouette['Silhouette Score']:.2f}. This suggests that this algorithm provides the best separation between clusters. "
                            f"On the other hand, {best_db['Algorithm']} performs best in terms of Davies-Bouldin Score with a score of {best_db['Davies-Bouldin Score']:.2f}, "
                            "indicating the lowest intra-cluster distances relative to the distances between clusters.")
                    
                    st.subheader("Detailed Results")
                    st.dataframe(results_df.style.background_gradient(cmap='RdYlGn', subset=['Silhouette Score'])
                                .background_gradient(cmap='RdYlGn_r', subset=['Davies-Bouldin Score']), use_container_width=True)
                    
                    st.write(f"**Insight:** Among all algorithms tested, the average Silhouette Score is {results_df['Silhouette Score'].mean():.2f}, "
                            f"and the average Davies-Bouldin Score is {results_df['Davies-Bouldin Score'].mean():.2f}. "
                            f"{'K-Means' if 'K-Means' in results_df['Algorithm'].values else 'Gaussian Mixture'} with different cluster numbers shows varying performance, "
                            "suggesting that the optimal number of clusters might depend on the specific characteristics of the startup data.")
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
                df_clustered = df.copy()
                df_clustered['Cluster'] = st.session_state['labels']
                df_clustered['Funding Amount in $'] = pd.to_numeric(df_clustered['Funding Amount in $'].replace('[\$,]', '', regex=True), errors='coerce')
                df_clustered['No. of Employees'] = df_clustered['No. of Employees'].apply(preprocess_employees)
                
                st.subheader("🎯 Key Clustering Insights")
                
                # Number of clusters
                n_clusters = len(df_clustered['Cluster'].unique())
                st.write(f"• The analysis identified {n_clusters} distinct clusters of startups.")
                
                # Funding distribution across clusters
                cluster_funding = df_clustered.groupby('Cluster')['Funding Amount in $'].mean().sort_values(ascending=False)
                top_cluster = cluster_funding.index[0]
                bottom_cluster = cluster_funding.index[-1]
                st.write(f"• Cluster {top_cluster} has the highest average funding (${cluster_funding[top_cluster]:,.0f}), "
                        f"while Cluster {bottom_cluster} has the lowest (${cluster_funding[bottom_cluster]:,.0f}).")
                
                # Employee distribution
                cluster_employees = df_clustered.groupby('Cluster')['No. of Employees'].mean().sort_values(ascending=False)
                top_emp_cluster = cluster_employees.index[0]
                st.write(f"• Cluster {top_emp_cluster} has the highest average number of employees ({cluster_employees[top_emp_cluster]:.0f}).")
                
                # Industry concentration
                top_industry_cluster = df_clustered.groupby('Cluster')['Industries'].apply(lambda x: ', '.join(x).split(', ')).apply(pd.Series).stack().value_counts().index[0]
                st.write(f"• The most common industry across all clusters is {top_industry_cluster}.")
                
                # Geographical patterns
                top_city_cluster = df_clustered.groupby('Cluster')['City'].value_counts().index[0]
                st.write(f"• {top_city_cluster[1]} has the highest representation in Cluster {top_city_cluster[0]}.")
                
                st.subheader("📊 Cluster Comparison")
                
                # Create a summary dataframe
                summary = df_clustered.groupby('Cluster').agg({
                    'Funding Amount in $': 'mean',
                    'No. of Employees': 'mean',
                    'Starting Year': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                    'No. of Investors': 'mean'
                }).round(2)
                
                st.dataframe(summary.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                
                # Highlight key differences
                oldest_cluster = summary['Starting Year'].idxmin()
                newest_cluster = summary['Starting Year'].idxmax()
                st.write(f"• Cluster {oldest_cluster} represents the oldest startups (most common year {summary.loc[oldest_cluster, 'Starting Year']:.0f}), "
                        f"while Cluster {newest_cluster} represents the newest (most common year {summary.loc[newest_cluster, 'Starting Year']:.0f}).")
                
                max_investor_cluster = summary['No. of Investors'].idxmax()
                st.write(f"• Cluster {max_investor_cluster} has attracted the most investors on average ({summary.loc[max_investor_cluster, 'No. of Investors']:.1f}).")
                
                st.subheader("🚀 Strategic Implications")
                st.write("""
                Based on these insights:
                - Investors might focus on startups in Cluster {top_cluster} for high-growth opportunities.
                - Entrepreneurs could aim to position their startups similarly to those in Cluster {top_emp_cluster} for rapid scaling.
                - Policymakers might consider initiatives to support diverse industry growth, given the concentration in {top_industry_cluster}.
                """.format(top_cluster=top_cluster, top_emp_cluster=top_emp_cluster, top_industry_cluster=top_industry_cluster))
            
            except Exception as e:
                st.error(f"An error occurred while generating insights: {str(e)}")
                st.write("Please check your data and try the clustering analysis again.")
