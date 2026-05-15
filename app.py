"""
Customer Segmentation - Interactive Streamlit Dashboard
Author: Raaga Priya JK
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="",
    layout="wide"
)

PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']
SEGMENT_NAMES = {
    0: 'Budget Families',
    1: 'Consistent Low-Spenders',
    2: 'Premium High-Value Customers',
    3: 'Campaign Responsive Customers'
}

# ── Load & cache data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    try:
        df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
        if df.shape[1] < 5:
            df = pd.read_csv('data/marketing_campaign.csv')
    except:
        df = pd.read_csv('data/marketing_campaign.csv')

    drop_cols = ['Z_CostContact', 'Z_Revenue', 'ID']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(subset=['Income'], inplace=True)
    df = df[df['Income'] < 200000]
    df = df[df['Year_Birth'] >= 1930]
    df['Marital_Status'] = df['Marital_Status'].replace(
        {'Absurd': 'Single', 'YOLO': 'Single', 'Alone': 'Single'})

    df['Age']              = 2024 - df['Year_Birth']
    df['TotalSpend']       = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                               df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['TotalPurchases']   = (df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'])
    df['TotalChildren']    = df['Kidhome'] + df['Teenhome']
    df['CampaignAccepted'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                  'AcceptedCmp4','AcceptedCmp5']].sum(axis=1)
    edu_map = {'Basic':0,'2n Cycle':1,'Graduation':2,'Master':3,'PhD':4}
    df['Edu_Encoded'] = df['Education'].map(edu_map)
    return df

@st.cache_data
def run_clustering(n_clusters, algorithm):
    df = load_and_clean()
    features = ['Income','Age','TotalSpend','TotalPurchases',
                'TotalChildren','CampaignAccepted','Recency','Edu_Encoded']
    X = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == 'K-Means++':
        model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)
    elif algorithm == 'Hierarchical (Ward)':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        df['Cluster'] = model.fit_predict(X_scaled)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=1.5, min_samples=10)
        df['Cluster'] = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df['Cluster']) if len(set(df['Cluster'])) > 1 else 0
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:,0], X_pca[:,1]
    return df, sil, pca.explained_variance_ratio_

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
algorithm  = st.sidebar.selectbox("Clustering Algorithm",
                                   ["K-Means++", "Hierarchical (Ward)", "DBSCAN"])
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Raaga Priya JK")
st.sidebar.markdown("**Dataset:** Customer Personality Analysis")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/raaga-priya/)")

# ── Main ───────────────────────────────────────────────────────────────────
st.title("Customer Segmentation & Targeting Strategy")
st.markdown("Interactive dashboard for K-Means++, Hierarchical, and DBSCAN clustering on Customer Personality Analysis data.")

df = load_and_clean()
df_clustered, sil_score, var_ratio = run_clustering(n_clusters, algorithm)

# ── Metrics row ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",    f"{len(df):,}")
col2.metric("Clusters",           n_clusters)
col3.metric("Silhouette Score",   f"{sil_score:.4f}")
col4.metric("PCA Variance (PC1+PC2)", f"{(var_ratio[0]+var_ratio[1])*100:.1f}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Clustering", "Cluster Profiles", "Business Insights"])

# ── Tab 1: EDA ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['Income'], bins=40, color='#457B9D', edgecolor='white', lw=0.5)
        ax.set_title('Income Distribution', fontweight='bold')
        ax.set_xlabel('Annual Income ($)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['TotalSpend'], bins=40, color='#2A9D8F', edgecolor='white', lw=0.5)
        ax.set_title('Total Spend Distribution', fontweight='bold')
        ax.set_xlabel('Total Spend ($)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(df['Income'], df['TotalSpend'], alpha=0.3,
                        c=df['Age'], cmap='plasma', s=10)
        ax.set_title('Income vs Total Spend (by Age)', fontweight='bold')
        ax.set_xlabel('Income ($)')
        ax.set_ylabel('Total Spend ($)')
        plt.colorbar(sc, ax=ax, label='Age')
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(7, 4))
        spend_cols = ['MntWines','MntMeatProducts','MntFishProducts',
                      'MntFruits','MntSweetProducts','MntGoldProds']
        spend_means = df[spend_cols].mean().sort_values()
        ax.barh(spend_means.index, spend_means.values, color='#E9C46A')
        ax.set_title('Avg Spend by Category', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ── Tab 2: Clustering ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Cluster Visualisation — {algorithm}")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in sorted(df_clustered['Cluster'].unique()):
            mask = df_clustered['Cluster'] == c
            label = SEGMENT_NAMES.get(c, f'Cluster {c}') if c != -1 else 'Noise'
            color = PALETTE[c % len(PALETTE)] if c != -1 else '#cccccc'
            ax.scatter(df_clustered.loc[mask,'PCA1'], df_clustered.loc[mask,'PCA2'],
                       c=color, label=label, alpha=0.5, s=15)
        ax.set_title('PCA Projection of Clusters', fontweight='bold')
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var)')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in sorted(df_clustered['Cluster'].unique()):
            mask = df_clustered['Cluster'] == c
            color = PALETTE[c % len(PALETTE)] if c != -1 else '#cccccc'
            ax.scatter(df_clustered.loc[mask,'Income'], df_clustered.loc[mask,'TotalSpend'],
                       c=color, label=f'C{c}', alpha=0.5, s=15)
        ax.set_title('Income vs Total Spend by Cluster', fontweight='bold')
        ax.set_xlabel('Annual Income ($)')
        ax.set_ylabel('Total Spend ($)')
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ── Tab 3: Cluster Profiles ────────────────────────────────────────────────
with tab3:
    st.subheader("Cluster Profile Summary")
    profile_cols = ['Income','Age','TotalSpend','TotalPurchases','TotalChildren','CampaignAccepted','Recency']
    profile = df_clustered[df_clustered['Cluster'] != -1].groupby('Cluster')[profile_cols].mean().round(1)
    profile['Count']   = df_clustered[df_clustered['Cluster'] != -1]['Cluster'].value_counts().sort_index()
    profile['Segment'] = [SEGMENT_NAMES.get(i, f'Cluster {i}') for i in profile.index]
    st.dataframe(profile, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ['Income','TotalSpend','TotalPurchases','CampaignAccepted']
    norm    = profile[metrics].copy()
    for col in metrics:
        norm[col] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)
    x = np.arange(len(metrics))
    w = 0.18
    for i, (idx, row) in enumerate(norm.iterrows()):
        ax.bar(x + i*w, row[metrics].values, width=w,
               color=PALETTE[i % len(PALETTE)],
               label=SEGMENT_NAMES.get(idx, f'C{idx}'), alpha=0.9)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(['Income', 'Total Spend', 'Purchases', 'Campaign Accepted'])
    ax.set_title('Normalised Feature Comparison by Cluster', fontweight='bold')
    ax.set_ylabel('Normalised Value (0-1)')
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close()

# ── Tab 4: Business Insights ───────────────────────────────────────────────
with tab4:
    st.subheader("Business Insights & Targeting Recommendations")

    insights = [
        {"Segment": "Premium High-Value Customers", "Cluster": "C2",
         "Strategy": "Loyalty rewards, early access, premium upsell",
         "Priority": "High"},
        {"Segment": "Campaign Responsive Customers", "Cluster": "C3",
         "Strategy": "Personalised offers, discount triggers, retargeting",
         "Priority": "High"},
        {"Segment": "Budget Families", "Cluster": "C0",
         "Strategy": "Value bundles, family deals, BNPL offers",
         "Priority": "Medium"},
        {"Segment": "Consistent Low-Spenders", "Cluster": "C1",
         "Strategy": "Re-engagement emails, time-limited incentives",
         "Priority": "Medium"},
    ]
    st.dataframe(pd.DataFrame(insights), use_container_width=True)

    st.markdown("### Key Takeaways")
    st.markdown("""
    - **Premium High-Value Customers (C2)** have the highest income and spend — prioritise retention
    - **Campaign Responsive Customers (C3)** are most likely to convert — ideal for targeted campaigns
    - **Budget Families (C0)** respond well to value offers and flexible payment options
    - **Consistent Low-Spenders (C1)** need re-engagement strategies to increase purchase frequency
    """)
