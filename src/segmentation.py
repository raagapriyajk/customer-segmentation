"""
Customer Segmentation & Targeting Strategy
===========================================
K-Means Clustering on Customer Personality Analysis Data

Author: Raaga Priya JK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']

# ── 1. Load Data ───────────────────────────────────────────────────────────
print("=" * 65)
print("CUSTOMER PERSONALITY ANALYSIS — SEGMENTATION PIPELINE")
print("=" * 65)

# Real Kaggle dataset is tab-separated
try:
    df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
    if df.shape[1] < 5:
        df = pd.read_csv('data/marketing_campaign.csv')
except:
    df = pd.read_csv('data/marketing_campaign.csv')

print(f"\n[1] Raw Dataset: {df.shape[0]} customers × {df.shape[1]} features")

# ── 2. Data Cleaning ───────────────────────────────────────────────────────
print("\n[2] Data Cleaning...")

before = len(df)

# Drop useless columns if they exist
drop_cols = ['Z_CostContact', 'Z_Revenue', 'ID']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Drop rows with missing Income
df.dropna(subset=['Income'], inplace=True)
print(f"   Removed {before - len(df)} rows with missing Income")

# Remove income outliers (above $200k — clearly data entry errors)
before2 = len(df)
df = df[df['Income'] < 200000]
print(f"   Removed {before2 - len(df)} income outlier rows (>$200k)")

# Remove age outliers (born before 1930 — unrealistic)
before3 = len(df)
df = df[df['Year_Birth'] >= 1930]
print(f"   Removed {before3 - len(df)} age outlier rows (born before 1930)")

# Fix weird Marital_Status values
df['Marital_Status'] = df['Marital_Status'].replace(
    {'Absurd': 'Single', 'YOLO': 'Single', 'Alone': 'Single'}
)
print(f"   Fixed Marital_Status: merged 'Absurd', 'YOLO', 'Alone' → 'Single'")

print(f"   ✓ Clean dataset: {len(df)} customers × {df.shape[1]} features")
print(df[['Income','MntWines','MntMeatProducts','Recency']].describe().round(1))

# ── 3. Feature Engineering ─────────────────────────────────────────────────
print("\n[3] Feature Engineering...")

df['Age']           = 2024 - df['Year_Birth']
df['TotalSpend']    = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                       df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
df['TotalPurchases']= (df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'])
df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
df['CampaignAccepted'] = (df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                               'AcceptedCmp4','AcceptedCmp5']].sum(axis=1))
df['SpendPerPurchase'] = np.where(df['TotalPurchases'] > 0,
                                   df['TotalSpend'] / df['TotalPurchases'], 0)

# Encode education
edu_map = {'Basic':0,'2n Cycle':1,'Graduation':2,'Master':3,'PhD':4}
df['Edu_Encoded'] = df['Education'].map(edu_map)

print(f"   New features: Age, TotalSpend, TotalPurchases, TotalChildren,")
print(f"                 CampaignAccepted, SpendPerPurchase, Edu_Encoded")

# ── 3. EDA ─────────────────────────────────────────────────────────────────
print("\n[3] EDA...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Personality Analysis — EDA', fontsize=16, fontweight='bold')

# Income distribution
axes[0,0].hist(df['Income'], bins=40, color='#457B9D', edgecolor='white', lw=0.5)
axes[0,0].set_title('Income Distribution', fontweight='bold')
axes[0,0].set_xlabel('Annual Income ($)')
axes[0,0].set_ylabel('Count')

# Total spend distribution
axes[0,1].hist(df['TotalSpend'], bins=40, color='#2A9D8F', edgecolor='white', lw=0.5)
axes[0,1].set_title('Total Spend Distribution', fontweight='bold')
axes[0,1].set_xlabel('Total Spend ($)')

# Spend by category
spend_cols = ['MntWines','MntMeatProducts','MntFishProducts',
              'MntFruits','MntSweetProducts','MntGoldProds']
spend_means = df[spend_cols].mean().sort_values(ascending=True)
axes[0,2].barh(spend_means.index, spend_means.values, color='#E9C46A')
axes[0,2].set_title('Avg Spend by Category', fontweight='bold')
axes[0,2].set_xlabel('Mean Spend ($)')

# Income vs Total Spend
sc = axes[1,0].scatter(df['Income'], df['TotalSpend'], alpha=0.3,
                        c=df['Age'], cmap='plasma', s=15, edgecolors='none')
axes[1,0].set_title('Income vs Total Spend (by Age)', fontweight='bold')
axes[1,0].set_xlabel('Income ($)')
axes[1,0].set_ylabel('Total Spend ($)')
plt.colorbar(sc, ax=axes[1,0], label='Age')

# Campaign acceptance
camp_total = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                 'AcceptedCmp4','AcceptedCmp5','Response']].sum()
axes[1,1].bar(camp_total.index, camp_total.values, color=PALETTE, width=0.6)
axes[1,1].set_title('Campaign Acceptance Count', fontweight='bold')
axes[1,1].set_ylabel('Customers Accepted')
axes[1,1].tick_params(axis='x', rotation=30)

# Education vs Total Spend
edu_spend = df.groupby('Education')['TotalSpend'].mean().sort_values()
axes[1,2].barh(edu_spend.index, edu_spend.values, color='#F4A261')
axes[1,2].set_title('Avg Spend by Education', fontweight='bold')
axes[1,2].set_xlabel('Avg Total Spend ($)')

plt.tight_layout()
plt.savefig('outputs/eda_overview.png', bbox_inches='tight')
plt.close()
print("   → Saved: outputs/eda_overview.png")

# ── 4. Correlation Heatmap ─────────────────────────────────────────────────
corr_cols = ['Income','Age','TotalSpend','TotalPurchases','TotalChildren',
             'CampaignAccepted','Recency','SpendPerPurchase']
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("   → Saved: outputs/correlation_heatmap.png")

# ── 5. Prepare Clustering Features ────────────────────────────────────────
print("\n[4] Preparing clustering features...")
cluster_features = ['Income','Age','TotalSpend','TotalPurchases',
                    'TotalChildren','CampaignAccepted','Recency','Edu_Encoded']

X = df[cluster_features].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Features: {cluster_features}")
print(f"   Shape: {X_scaled.shape}")

# ── 6. Elbow + Silhouette ──────────────────────────────────────────────────
print("\n[5] Finding optimal K...")
inertias, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = list(K_range)[sil_scores.index(max(sil_scores))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Optimal K Selection', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'o-', color='#457B9D', lw=2, ms=7)
axes[0].axvline(x=4, color='#E63946', ls='--', alpha=0.7, label='Optimal K=4')
axes[0].set_title('Elbow Method (Inertia)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()

axes[1].plot(list(K_range), sil_scores, 's-', color='#2A9D8F', lw=2, ms=7)
axes[1].axvline(x=best_k, color='#E63946', ls='--', alpha=0.7, label=f'Best K={best_k}')
axes[1].set_title('Silhouette Score', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/elbow_silhouette.png', bbox_inches='tight')
plt.close()
print(f"   → Best K by silhouette: {best_k} (score: {max(sil_scores):.3f})")

# ── 7. Final K-Means ──────────────────────────────────────────────────────
OPTIMAL_K = 4
print(f"\n[6] K-Means clustering (K={OPTIMAL_K})...")
kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=20, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, df['Cluster'])
print(f"   Silhouette Score: {final_sil:.4f}")
print(f"   Cluster sizes:\n{df['Cluster'].value_counts().sort_index().to_string()}")

# ── 8. PCA Visualization ───────────────────────────────────────────────────
print("\n[7] Cluster visualizations...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('K-Means Segmentation Results', fontsize=14, fontweight='bold')

for c in range(OPTIMAL_K):
    mask = df['Cluster'] == c
    axes[0].scatter(df.loc[mask,'PCA1'], df.loc[mask,'PCA2'],
                    c=PALETTE[c], label=f'Cluster {c}',
                    alpha=0.5, s=20, edgecolors='none')
centres_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(centres_pca[:,0], centres_pca[:,1], c='black',
                marker='X', s=200, zorder=5, label='Centroids')
axes[0].set_title('PCA Projection of Clusters', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
axes[0].legend(markerscale=1.5)

for c in range(OPTIMAL_K):
    mask = df['Cluster'] == c
    axes[1].scatter(df.loc[mask,'Income'], df.loc[mask,'TotalSpend'],
                    c=PALETTE[c], label=f'Cluster {c}',
                    alpha=0.5, s=20, edgecolors='none')
axes[1].set_title('Income vs Total Spend by Cluster', fontweight='bold')
axes[1].set_xlabel('Annual Income ($)')
axes[1].set_ylabel('Total Spend ($)')
axes[1].legend(markerscale=1.5)

plt.tight_layout()
plt.savefig('outputs/cluster_visualization.png', bbox_inches='tight')
plt.close()
print("   → Saved: outputs/cluster_visualization.png")

# ── 9. Cluster Profile Analysis ────────────────────────────────────────────
print("\n[8] Cluster profiling...")
profile_cols = ['Income','Age','TotalSpend','TotalPurchases',
                'TotalChildren','CampaignAccepted','Recency']
profile = df.groupby('Cluster')[profile_cols].mean().round(1)
profile['Count']   = df['Cluster'].value_counts().sort_index()
profile['% Share'] = (profile['Count'] / len(df) * 100).round(1)

SEGMENT_NAMES = {
    0: 'Budget Families',
    1: 'High-Value Loyalists',
    2: 'At-Risk Customers',
    3: 'Potential Converters',
}
profile['Segment'] = [SEGMENT_NAMES[i] for i in profile.index]
print(profile.to_string())

# Radar-style bar chart
fig, ax = plt.subplots(figsize=(14, 6))
metrics    = ['Income','TotalSpend','TotalPurchases','CampaignAccepted']
bar_w      = 0.18
x          = np.arange(len(metrics))
norm_profile = profile[metrics].copy()
for col in metrics:
    norm_profile[col] = (norm_profile[col] - norm_profile[col].min()) / \
                        (norm_profile[col].max() - norm_profile[col].min() + 1e-9)

for i in range(OPTIMAL_K):
    vals = norm_profile.iloc[i][metrics].values
    ax.bar(x + i * bar_w, vals, width=bar_w, color=PALETTE[i],
           label=f"C{i}: {SEGMENT_NAMES[i]}", alpha=0.9)

ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(['Income', 'Total Spend', 'Purchases', 'Campaigns\nAccepted'])
ax.set_title('Normalised Cluster Feature Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalised Mean Value (0–1)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/cluster_profiles.png', bbox_inches='tight')
plt.close()
print("   → Saved: outputs/cluster_profiles.png")

# ── 10. Campaign Response Analysis ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Campaign Response by Cluster', fontsize=14, fontweight='bold')

camp_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']
camp_rate = df.groupby('Cluster')[camp_cols].mean() * 100

for c in range(OPTIMAL_K):
    axes[0].plot(camp_rate.columns, camp_rate.loc[c],
                 'o-', color=PALETTE[c], label=f'C{c}: {SEGMENT_NAMES[c]}', lw=2, ms=6)
axes[0].set_title('Campaign Acceptance Rate (%)', fontweight='bold')
axes[0].set_ylabel('Acceptance Rate (%)')
axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=20)

spend_by_cluster = df.groupby('Cluster')[['MntWines','MntMeatProducts',
                               'MntFruits','MntSweetProducts','MntGoldProds']].mean()
spend_by_cluster.T.plot(kind='bar', ax=axes[1], color=PALETTE[:OPTIMAL_K],
                         width=0.7, edgecolor='white')
axes[1].set_title('Avg Category Spend by Cluster', fontweight='bold')
axes[1].set_ylabel('Avg Spend ($)')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend([f'C{i}' for i in range(OPTIMAL_K)], fontsize=8)

plt.tight_layout()
plt.savefig('outputs/campaign_analysis.png', bbox_inches='tight')
plt.close()
print("   → Saved: outputs/campaign_analysis.png")

# ── 11. Business Insights ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("BUSINESS INSIGHTS & TARGETING RECOMMENDATIONS")
print("=" * 65)

insights = {
    'High-Value Loyalists':  'High income + high spend + campaign responsive → Premium loyalty rewards, early access offers',
    'Potential Converters':  'Moderate income, low spend → Personalised win-back campaigns, discount triggers',
    'Budget Families':       'Lower income, kids at home → Value bundles, family deals, BNPL offers',
    'At-Risk Customers':     'High recency (inactive) → Re-engagement emails, targeted incentives',
}
for seg, rec in insights.items():
    print(f"  • {seg:22s}: {rec}")

# Save outputs
profile.to_csv('outputs/cluster_summary.csv')
df.to_csv('outputs/segmented_customers.csv', index=False)
print(f"\n[✓] Outputs saved to outputs/")
print(f"[✓] Total customers segmented: {len(df)}")
print(f"[✓] Final Silhouette Score: {final_sil:.4f}")
