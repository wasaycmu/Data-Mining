# Data-Mining

## 1. Problem Description

### Business Understanding

This project aims to identify which population segments or geographic regions in the U.S. should be prioritized for interventions in the event of another pandemic like COVID-19. The stakeholder is a public health policy planner or government agency seeking to optimize the allocation of healthcare resources and design preventive strategies.

### Key guiding questions:
- Which demographic or socioeconomic groups are most associated with higher COVID-19 cases and deaths?
- Do health-related behaviors or access (e.g., smoking, healthcare quality, vaccination rates) significantly affect COVID-19 outcomes?
- What role do age, education, income, and ethnicity play?
- Are urban, dense, or transit-heavy areas inherently more at risk?

### Why it matters
Understanding these relationships allows for proactive and targeted policy responses, rather than blanket approaches that waste resources and fail to protect the most vulnerable.

### Relevant Data Needed:
- **COVID-19 case and death counts**
- **Demographics** (age, race, income, education)
- **Health indicators** (smoking, obesity, pre-existing conditions)
- **Socioeconomic variables** (poverty, income ratios)
- **Public health responses** (lockdown timing, vaccination rates)
- **Geographical variables** (urban vs. rural, population density)

## 2. Data Collection and Data Quality

The data was aggregated from multiple reputable sources at the county level for Texas, including the U.S. Census, CDC, and health surveys. All relevant fields were normalized per thousand population for comparability. Variables were cleaned and labeled clearly for analysis. The final dataset contains counties as rows and demographic, health, and socioeconomic indicators as columns.

Merging was done via county FIPS codes. The data quality is high, but gaps exist in less-populated counties. Missing or inconsistent entries were excluded where appropriate.

## 3. Data Exploration

### Key Descriptive Findings:
- **Population Density:** Weak correlation with cases (+0.02) and a moderate negative correlation with deaths (-0.24). High-density areas may fare better in death rates due to better healthcare infrastructure.
- **Smokers and Median Age:** Higher smoking prevalence and older median ages are found in less dense counties. Younger, denser areas (urban centers) show healthier behaviors.
- **Health Status:** Counties with a higher percentage of residents in fair or poor health have significantly worse COVID-19 outcomes.
- **Vaccination:** Strong negative correlation with deaths (-0.35), confirming vaccines reduce mortality.
- **Income and Education:** Higher median income and higher education levels correlate with lower cases and deaths.
- **Ethnicity:** Hispanic populations show a strong positive correlation with both cases and deaths, pointing to systemic disparities.
- **Transit and Children:** Public transit use and family structure (young children) showed negligible effects on case and death rates and were deprioritized.
- **Lockdown Effectiveness:** Weak but positive correlation with reduced deaths (0.22), suggesting benefit.

### Correlation Matrix Highlights (Deaths & Cases):
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

### 3.1 Modeling and Evaluation

Initial modeling involved correlation matrices to assess linear relationships. Based on the results, we recommend focusing further modeling efforts (e.g., multiple linear regression or random forest classification) on:

- Median income
- Education level
- Health status
- Vaccination rates
- Ethnic distribution

Variables like population density, children, and transit options were dropped due to their weak associations.

## 4. Recommendations

### For Policy & Public Health Planning:
- **Target Health Disparities:** Prioritize counties with a high percentage of residents in fair/poor health for medical resources and health education.
- **Support Hispanic Communities:** Investigate and address systemic barriers in areas with high Hispanic populations.
- **Boost Vaccination:** Expand outreach and access, especially in counties with low vaccination uptake.
- **Focus on Education:** Invest in long-term health literacy and public health campaigns tied to educational attainment.
- **Deprioritize High-Density Focus:** Urban density itself does not strongly correlate with higher COVID-19 deaths and should not be the main determinant for resource allocation.

## 5. Conclusion

Yes, the project answered the original question. Not all assumptions held up—population density and transit use were weakly related to COVID-19 outcomes. Instead, demographic health indicators (e.g., smoking, income, ethnicity, education, and vaccination) emerged as far stronger predictors. These findings are critical in planning better-targeted responses for future pandemics.

## 6. List of References

- CDC COVID Data Tracker. (2020). [https://covid.cdc.gov/covid-data-tracker/](https://covid.cdc.gov/covid-data-tracker/)
- U.S. Census Bureau. (2020). [https://www.census.gov](https://www.census.gov)
- County Health Rankings & Roadmaps. [https://www.countyhealthrankings.org/](https://www.countyhealthrankings.org/)
- Texas Department of State Health Services. [https://www.dshs.texas.gov/](https://www.dshs.texas.gov/)
- Public Health Reports, peer-reviewed literature (various)
