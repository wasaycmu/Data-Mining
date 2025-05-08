# COVID-19 Data Mining Projects

## 🧠 Techniques Used

- Data Cleaning & Normalization  
- Exploratory Data Analysis (EDA)  
- Correlation Analysis  
- Feature Engineering  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  
- Neural Networks  
- Model Evaluation (Accuracy, Kappa, ARI, NMI, Silhouette Score)  
- Hyperparameter Tuning (Grid Search)  
- Clustering (DBSCAN, Model-Based Clustering, K-Means, Hierarchical)  
- Risk Stratification & Quantile Classification  

## 🛠 Technologies Used

- **R** (Primary language across all projects)
  - `tidyverse`, `ggplot2`, `dplyr`, `caret`, `randomForest`, `e1071`, `xgboost`, `nnet`, `mclust`, `dbscan`, `cluster`, `factoextra`, etc.
- CSV & county-level public datasets
- GIS-style geographic analysis via FIPS codes

---

## 📁 Project 1: COVID-19 Demographic Risk Analysis

### Objective
Identify which demographic or geographic regions in the U.S. (Texas counties) should be prioritized for interventions in future pandemics.

### Key Questions Addressed
- Which groups are more vulnerable to COVID-19?
- Do health behaviors or healthcare access impact outcomes?
- Are urban/dense areas truly more at risk?

### Dataset Details
- Source: CDC, U.S. Census, Texas Health Services
- Granularity: County-level
- Features: Demographics, income, health status, vaccination rates, etc.

### Findings
- **Income, education, health status, and vaccination** are strongly correlated with COVID-19 outcomes.
- **Population density** and **public transit usage** were weakly correlated.
- Hispanic populations faced disproportionate risks.

### Correlation Matrix Highlights

| Factor                  | Corr. with Deaths | Corr. with Cases |
|------------------------|-------------------|------------------|
| Median Income          | -0.40             | -0.13            |
| Vaccination Rate       | -0.35             | -0.17            |
| % in Poor Health       | +0.39             | +0.42            |
| % Hispanic Population  | +0.31             | +0.47            |

---

## 📁 Project 2: COVID-19 County Clustering (DBSCAN & Model-Based)

### Objective
Cluster Texas counties by COVID-19 outcomes and socio-demographic features to identify similar patterns and outlier regions.

### Methods Used
- **DBSCAN (Density-Based Spatial Clustering)**
- **Model-Based Clustering** (Gaussian Mixture Models)
- **K-Means & Hierarchical Clustering** for benchmarking
- **Evaluation Metrics**: Silhouette Score, ARI, NMI

### Results Summary

- **DBSCAN** revealed outlier counties with poor outcomes and low vaccination.
- **Model-Based Clustering** produced interpretable clusters; best-performing group had high income and vaccination.
- **K-Means** outperformed other methods on ARI/NMI metrics for supervised labels.

### Key Cluster Insights
- High-Hispanic and low-income clusters had worse outcomes.
- Outliers detected by DBSCAN deserve targeted policy focus.
- Clustering can supplement traditional classification for resource prioritization.

---

## 📁 Project 3: COVID-19 Risk Classification with ML

### Objective
Predict COVID-19 risk levels (low, medium, high) at the county level using supervised classification models.

### Models Used
- Logistic Regression
- Random Forest
- SVM
- XGBoost
- Neural Network

### Feature Engineering
- `health_risk_score`: Composite of obesity, smoking, poor health
- `socioeconomic_score`: Composite of poverty and unemployment

### Results Summary

| Model           | Accuracy | Kappa   | Notes                                  |
|----------------|----------|---------|----------------------------------------|
| Random Forest  | 57.14%   | 0.3553  | Best overall performer                 |
| XGBoost        | ~56%     | ~0.34   | Good for feature ranking               |
| Others         | <55%     | Low     | Overlap in classes limited performance |

### Challenges
- Quantile-based class definitions caused overlap and reduced class separability.

### Recommendations
- Redefine class boundaries
- Use ensemble and stacked models
- Integrate finer-grain features like hospital access

---

## 📌 Overall Conclusions

- Socioeconomic and health disparity indicators far outweigh geographic factors like density or transit access.
- Clustering methods like DBSCAN are useful for identifying at-risk outliers.
- Classification models show moderate performance but provide actionable directionality.
- Strong case for targeted intervention strategies based on education, vaccination, and health status rather than population size or urban/rural designation.

---

## 📚 References

- CDC COVID Data Tracker. (2020). https://covid.cdc.gov/covid-data-tracker/  
- U.S. Census Bureau. https://www.census.gov  
- County Health Rankings. https://www.countyhealthrankings.org/  
- Texas DSHS. https://www.dshs.texas.gov/  
- Peer-reviewed literature and public health reports  
