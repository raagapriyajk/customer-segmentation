# 🛍️ Customer Segmentation & Targeting Strategy

> **K-Means Clustering on Customer Personality Analysis data to identify high-value segments and drive targeted marketing campaigns**

---

## 📌 Project Overview

This project applies **unsupervised machine learning** on a rich customer personality dataset to segment customers based on their **demographics, spending behaviour, purchase patterns, and campaign responsiveness**. The goal is to help marketing teams identify distinct customer profiles and design data-driven targeting strategies for each group.

---

## 🎯 Objectives

- Perform in-depth **Exploratory Data Analysis (EDA)** on customer demographics and behaviour
- Engineer meaningful features from raw transactional and campaign data
- Use **Elbow Method & Silhouette Score** to determine the optimal number of clusters
- Build a **K-Means clustering model** to segment 2,240 customers into 4 actionable groups
- Derive **business insights** and targeting recommendations per segment
- Analyse **campaign response rates** across segments to prioritise marketing spend

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Stats | scikit-learn, NumPy |
| Data Processing | Pandas |
| Visualisation | Matplotlib, Seaborn |
| Dimensionality Reduction | PCA |

---

## 📂 Project Structure

```
customer-segmentation/
│
├── data/
│   ├── generate_data.py          # Synthetic data generation script
│   └── marketing_campaign.csv    # Customer Personality dataset (2,240 records)
│
├── src/
│   └── segmentation.py           # Full ML pipeline
│
├── outputs/
│   ├── eda_overview.png           # EDA charts
│   ├── correlation_heatmap.png    # Feature correlations
│   ├── elbow_silhouette.png       # K selection plots
│   ├── cluster_visualization.png  # PCA + Income vs Spend plots
│   ├── cluster_profiles.png       # Normalised cluster comparison
│   ├── campaign_analysis.png      # Campaign response by cluster
│   ├── cluster_summary.csv        # Cluster mean stats
│   └── segmented_customers.csv    # Full dataset with cluster labels
│
├── requirements.txt
└── README.md
```

---

## 🔍 Features Used for Clustering

| Feature | Description |
|---|---|
| `Income` | Annual household income |
| `Age` | Derived from Year_Birth |
| `TotalSpend` | Sum across all product categories |
| `TotalPurchases` | Web + Catalog + Store purchases |
| `TotalChildren` | Kids + Teens at home |
| `CampaignAccepted` | Total campaigns accepted (1–5) |
| `Recency` | Days since last purchase |
| `Edu_Encoded` | Education level (ordinal encoded) |

---

## 📊 Cluster Results (K = 4)

| Cluster | Segment | Avg Income | Avg Spend | Key Trait |
|---------|---------|-----------|-----------|-----------|
| C0 | Budget Families | ~$50k | ~$649 | Moderate income, low campaign response |
| C1 | High-Value Loyalists | ~$29k | ~$368 | Lower income but consistent spenders |
| C2 | At-Risk Customers | ~$71k | ~$933 | High income & spend, campaign active |
| C3 | Potential Converters | ~$49k | ~$624 | Highest campaign acceptance rate |

---

## 💡 Business Insights

| Segment | Recommendation |
|---------|---------------|
| **High-Value Loyalists** | Premium loyalty rewards, early access, exclusive bundles |
| **Potential Converters** | Personalised win-back campaigns, discount triggers, retargeting |
| **Budget Families** | Value bundles, family deals, BNPL offers |
| **At-Risk Customers** | Re-engagement emails, time-limited incentives |

---

## 🔁 Methodology

```
Raw Data (2,240 rows × 26 cols)
  → Feature Engineering (Age, TotalSpend, TotalPurchases, CampaignAccepted)
    → StandardScaler normalisation
      → Elbow Method + Silhouette Score → K = 4
        → K-Means++ Clustering
          → PCA Visualisation
            → Cluster Profiling + Business Insights
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/raagapriya/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_data.py

# Run full analysis
python src/segmentation.py
```

All output plots and CSVs will be saved in the `outputs/` folder.

---

## 👤 Author

**Raaga Priya JK**
- 📧 raagapriya.jk28@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/raaga-priya/)
- B.Tech AI & Data Science — Sri Venkateswara College of Engineering
