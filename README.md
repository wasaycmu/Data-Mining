# COVID-19 Data Mining Projects

## 1. Data-Mining

### Problem Description

#### Business Understanding
This project aims to identify which population segments or geographic regions in the U.S. should be prioritized for interventions in the event of another pandemic like COVID-19. The stakeholder is a public health policy planner or government agency seeking to optimize the allocation of healthcare resources and design preventive strategies.

#### Key guiding questions:
- Which demographic or socioeconomic groups are most associated with higher COVID-19 cases and deaths?
- Do health-related behaviors or access (e.g., smoking, healthcare quality, vaccination rates) significantly affect COVID-19 outcomes?
- What role do age, education, income, and ethnicity play?
- Are urban, dense, or transit-heavy areas inherently more at risk?

#### Why it matters
Understanding these relationships allows for proactive and targeted policy responses, rather than blanket approaches that waste resources and fail to protect the most vulnerable.

#### Relevant Data Needed:
- **COVID-19 case and death counts**
- **Demographics** (age, race, income, education)
- **Health indicators** (smoking, obesity, pre-existing conditions)
- **Socioeconomic variables** (poverty, income ratios)
- **Public health responses** (lockdown timing, vaccination rates)
- **Geographical variables** (urban vs. rural, population density)

### Data Collection and Data Quality

The data was aggregated from multiple reputable sources at the county level for Texas, including the U.S. Census, CDC, and health surveys. All relevant fields were normalized per thousand population for comparability. Variables were cleaned and labeled clearly for analysis. The final dataset contains counties as rows and demographic, health, and socioeconomic indicators as columns.

Merging was done via county FIPS codes. The data quality is high, but gaps exist in less-populated counties. Missing or inconsistent entries were excluded where appropriate.

### Data Exploration

#### Key Descriptive Findings:
- **Population Density:** Weak correlation with cases (+0.02) and a moderate negative correlation with deaths (-0.24). High-density areas may fare better in death rates due to better healthcare infrastructure.
- **Smokers and Median Age:** Higher smoking prevalence and older median ages are found in less dense counties. Younger, denser areas (urban centers) show healthier behaviors.
- **Health Status:** Counties with a higher percentage of residents in fair or poor health have significantly worse COVID-19 outcomes.
- **Vaccination:** Strong negative correlation with deaths (-0.35), confirming vaccines reduce mortality.
- **Income and Education:** Higher median income and higher education levels correlate with lower cases and deaths.
- **Ethnicity:** Hispanic populations show a strong positive correlation with both cases and deaths, pointing to systemic disparities.
- **Transit and Children:** Public transit use and family structure (young children) showed negligible effects on case and death rates and were deprioritized.
- **Lockdown Effectiveness:** Weak but positive correlation with reduced deaths (0.22), suggesting benefit.

#### Correlation Matrix Highlights (Deaths & Cases):
| Factor                    | Correlation with Deaths | Correlation with Cases |
|---------------------------|-------------------------|------------------------|
| Median Income              | -0.40                   | -0.13                  |
| Vaccination Rate           | -0.35                   | -0.17                  |
| % in Poor Health           | +0.39                   | +0.42                  |
| % Hispanic Population      | +0.31                   | +0.47                  |
| % Smokers                  | +0.21                   | +0.19                  |
| Median Age                 | -0.07                   | -0.41                  |
| Population Density         | -0.24                   | +0.02                  |
| Public Transit Use         | -0.18                   | +0.04                  |

#### 3.1 Modeling and Evaluation

Initial modeling involved correlation matrices to assess linear relationships. Based on the results, we recommend focusing further modeling efforts (e.g., multiple linear regression or random forest classification) on:
- Median income
- Education level
- Health status
- Vaccination rates
- Ethnic distribution

Variables like population density, children, and transit options were dropped due to their weak associations.

### Recommendations

#### For Policy & Public Health Planning:
- **Target Health Disparities:** Prioritize counties with a high percentage of residents in fair/poor health for medical resources and health education.
- **Support Hispanic Communities:** Investigate and address systemic barriers in areas with high Hispanic populations.
- **Boost Vaccination:** Expand outreach and access, especially in counties with low vaccination uptake.
- **Focus on Education:** Invest in long-term health literacy and public health campaigns tied to educational attainment.
- **Deprioritize High-Density Focus:** Urban density itself does not strongly correlate with higher COVID-19 deaths and should not be the main determinant for resource allocation.

### Conclusion
Yes, the project answered the original question. Not all assumptions held up—population density and transit use were weakly related to COVID-19 outcomes. Instead, demographic health indicators (e.g., smoking, income, ethnicity, education, and vaccination) emerged as far stronger predictors. These findings are critical in planning better-targeted responses for future pandemics.

### List of References
- CDC COVID Data Tracker. (2020). [https://covid.cdc.gov/covid-data-tracker/](https://covid.cdc.gov/covid-data-tracker/)
- U.S. Census Bureau. (2020). [https://www.census.gov](https://www.census.gov)
- County Health Rankings & Roadmaps. [https://www.countyhealthrankings.org/](https://www.countyhealthrankings.org/)
- Texas Department of State Health Services. [https://www.dshs.texas.gov/](https://www.dshs.texas.gov/)
- Public Health Reports, peer-reviewed literature (various)

---

## 2. Data-Mining-2: COVID-19 Clustering Analysis: DBSCAN & Model-Based Clustering

This project applies unsupervised learning techniques — specifically DBSCAN and Model-Based Clustering — to categorize Texas counties based on COVID-19 outcomes and key demographic, socioeconomic, and health-related variables.

### Project Objective
To identify meaningful groupings of counties with similar COVID-19 profiles — including outlier detection — in order to guide targeted public health interventions and support data-driven policymaking.

### Clustering Methods Overview

#### 1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **How It Works:** Groups closely packed points and marks points in sparse regions as outliers.
- **Why Use It:** No need to predefine the number of clusters. Handles non-spherical clusters and noise effectively.
- **Use Case:** Distinguishes densely impacted counties and flags outliers needing special focus.

#### DBSCAN Results
| Cluster | # Counties | Avg. Pop | Cases/1K | Deaths/1K | % Vaccinated | Median Income | % Below Poverty |
|---------|------------|----------|----------|-----------|--------------|---------------|-----------------|
| 1       | 203        | ~89,177  | ~77.63   | ~1.85     | ~36.63%      | ~$49,500      | ~16.76%         |
| 0 (Noise) | 49       | ~193,688 | ~80.09   | ~1.95     | ~35.63%      | ~$51,373      | ~16.92%         |

#### Interpretation:
- **Cluster 1:** Moderately populated counties with slightly better COVID-19 outcomes.
- **Cluster 0 (Noise):** Larger counties, likely urban/suburban, but with low vaccination rates and worse outcomes.

#### 2. Model-Based Clustering
- **How It Works:** Assumes data is from a mixture of probability distributions (e.g., Gaussian).
- **Why Use It:** Automatically determines the optimal number of clusters (via BIC). Supports soft clustering and probabilistic interpretations.

#### Model-Based Clustering Results
| Cluster | # Counties | Pop | Cases/1K | Deaths/1K | % Vaccinated | Median Income | Insight |
|---------|------------|-----|----------|-----------|--------------|---------------|---------|
| 1       | 18         | ~436,587 | ~77.2   | ~1.41     | ~42.5%      | ~$56,540      | Best outcomes: High vaccination/income |
| 2       | 103        | ~36,006  | ~75.9   | ~2.04     | ~32.3%      | ~$49,788      | Worst outcomes: Rural, low access |
| 3       | 100        | ~60,962  | ~81.3   | ~1.85     | ~37.4%      | ~$48,434      | High cases, moderate outcomes |
| 4       | 31         | ~316,898 | ~75.5   | ~1.61     | ~43.4%      | ~$50,873      | Suburban counties doing well |

### Clustering Evaluation

- **DBSCAN:** 0.244 Silhouette Score
- **K-Means:** 0.215 Silhouette Score
- **Hierarchical:** 0.179 Silhouette Score
- **Model-Based:** 0.069 Silhouette Score

#### Supervised Evaluation (Death Rate Categories)
- **Metrics Used:** Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)

#### Evaluation Summary
- **K-Means:** Highest ARI and NMI, clear alignment with actual death rates.
- **Hierarchical:** Reveals nested risk structure.
- **DBSCAN:** Excellent for finding outlier counties.

### Key Insights
- **Hispanic Population:** High-risk clusters had higher Hispanic populations. Focus healthcare access and public messaging in these areas.
- **Income Disparity:** Low-income counties had higher death rates. Increase economic and healthcare support in low-income counties.
- **Vaccination Rate:** Correlates with lower death rates. Sustain vaccination drives, especially in rural areas.
- **Health & Education:** Better education and health indicators led to lower mortality. Long-term investment in education and healthcare is needed.

### Stakeholder Recommendations
- Boost Vaccination: Prioritize underperforming clusters.
- Target Outliers: Flag special-focus counties using DBSCAN’s noise detection.
- Focus on Equity: Address poverty, access, and education gaps in policy.

---

## 3. Data-Mining-3: Classification Analysis on COVID-19 Data

### Overview
This project classifies COVID-19 risk levels (low, medium, high) for counties in Texas using demographic, socioeconomic, and health-related data. Several machine learning models were tested, including Logistic Regression, Random Forest, SVM, XGBoost, and Neural Networks.

### Executive Summary
- **Objective:** Predict county-level COVID-19 risk levels using quantile-based classification.
- **Top Models:** Random Forest and XGBoost performed best with ~57.14% accuracy.
- **Insight:** Quantile-based class definitions created overlap, reducing model separability but offering directional insights for stakeholders.

### Business Understanding
The project helps understand how COVID-19 impacted different populations and builds predictive models to assist public health stakeholders in planning and resource allocation.

### Data Preparation
The dataset includes demographic, socioeconomic, and health features at the county level, with a clean dataset and no missing values.

### Modeling Approach

#### Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Neural Network

### Key Metrics Summary
| Model | Accuracy | Kappa | Best Use Case |
|-------|----------|-------|---------------|
| Random Forest | 57.14% | 0.3553 | Balanced accuracy, feature importance |

### Feature Engineering
- **health_risk_score:** Avg of poor health, obesity, smoking
- **socioeconomic_score:** Avg of unemployment and poverty

### Hyperparameter Tuning
Best hyperparameters were found using grid search.

### Recommendations
- Redefine class boundaries beyond quantiles.
- Explore ensemble and stacking methods.
- Collect more granular features (e.g., healthcare infrastructure).

