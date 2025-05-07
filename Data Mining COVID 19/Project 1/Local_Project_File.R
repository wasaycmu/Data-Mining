library("tidyverse")
library("ggplot2")
library(readr)
library(dplyr)
library(missForest)
library("ggrepel")
library("ggcorrplot")
library("DT")
library("tidyr")
library(gridExtra)
library(purrr)
library(ggrepel)
library(sf)
library(corrplot)
library(knitr) 

##READ ALL FILES

census_data <- read.csv("COVID-19_cases_plus_census.csv")
tx_data <- read.csv("COVID-19_cases_TX.csv")
gmr_data <- read.csv("gmr_tx.csv")
vacc_data <- read.csv("Texas Vaccine Data by County.csv")
health_data <- read.csv("US_counties_COVID19_health_weather_data.csv")

#INITIAL LOOK AT FILES
str(census_data, list.len=ncol(census_data))
str(tx_data, list.len=ncol(tx_data))
str(gmr_data, list.len=ncol(gmr_data))
str(vacc_data, list.len=ncol(vacc_data))
str(health_data, list.len=ncol(health_data))
str(geometric_data, list.len=ncol(geometric_data))
str(socio_health_data, list.len=ncol(socio_health_data))


#QUESTIONS TO BE ANSWERED

#1 population density and urbanization
#2 transit options (train vs cars)
#3 ethnic relation 
#4 income
#5 lockdown
#6 age
#7 children increase risk
#8 education levels
#9 vaccination 
#10 health factors



#DATA QUALITY TESTING FOR CENSUS DATASET

tx_data <- census_data %>%
  filter(state == "TX")  

# Specify columns to analyze
columns_to_analyze <- c(
  "total_pop",
  "county_fips_code",
  "commuters_by_public_transportation",
  "commuters_by_car_truck_van",
  "commuters_by_subway_or_elevated",
  "white_pop",
  "black_pop",
  "asian_pop",
  "hispanic_pop",
  "amerindian_pop",
  "other_race_pop",
  "two_or_more_races_pop",
  "not_hispanic_pop",
  "median_income",
  "median_age",
  "families_with_young_children",
  "high_school_diploma",
  "bachelors_degree",
  "graduate_professional_degree",
  "bachelors_degree_or_higher_25_64"
)

# Initialize result data frame
results <- data.frame()

# Function to analyze each column
analyze_column <- function(column_name) {
  column_data <- tx_data[[column_name]]
  
  # Calculate metrics
  total_values <- length(column_data)
  na_count <- sum(is.na(column_data))
  negative_count <- sum(column_data < 0, na.rm = TRUE)
  duplicate_count <- nrow(tx_data) - nrow(distinct(tx_data, !!sym(column_name)))
  
  # Identify outliers using IQR method
  Q1 <- quantile(column_data, 0.25, na.rm = TRUE)
  Q3 <- quantile(column_data, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  outlier_threshold_high <- Q3 + 1.5 * IQR
  outlier_threshold_low <- Q1 - 1.5 * IQR
  outliers <- sum(column_data < outlier_threshold_low | column_data > outlier_threshold_high, na.rm = TRUE)
  
  percentage_outliers <- (outliers / total_values) * 100
  
  # Determine variable type
  variable_type <- class(column_data)
  
  # Create result row
  result_row <- data.frame(
    Variable = column_name,
    Type = variable_type,
    NAs = na_count,
    Percentage_Outliers = percentage_outliers,
    Negative_Values = negative_count,
    Duplicate_Values = duplicate_count,
    Total_Values = total_values
  )
  
  return(result_row)
}

# Loop through each column and analyze
for (column in columns_to_analyze) {
  results <- rbind(results, analyze_column(column))
}

# Display results
print(results)

write.csv(results, "census_analysis_results.csv", row.names = FALSE)

#DATA QUALITY TESTING FOR GMR DATASET
texas_data <- gmr_data %>%
  filter(sub_region_1 == "Texas")  

# Specify columns to analyze
columns_to_analyze <- c(
  "sub_region_1",
  "sub_region_2",
  "metro_area",
  "census_fips_code",
  "transit_stations_percent_change_from_baseline",
  "date",
  "workplaces_percent_change_from_baseline"
)

# Initialize result data frame
results <- data.frame()

# Function to analyze each column
analyze_column <- function(column_name) {
  column_data <- texas_data[[column_name]]
  
  # Calculate metrics
  total_values <- length(column_data)
  na_count <- sum(is.na(column_data))
  negative_count <- sum(column_data < 0, na.rm = TRUE)
  duplicate_count <- nrow(texas_data) - nrow(distinct(texas_data, !!sym(column_name)))
  
  # Identify outliers using IQR method (only for numeric columns)
  if (is.numeric(column_data)) {
    Q1 <- quantile(column_data, 0.25, na.rm = TRUE)
    Q3 <- quantile(column_data, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    outlier_threshold_high <- Q3 + 1.5 * IQR
    outlier_threshold_low <- Q1 - 1.5 * IQR
    outliers <- sum(column_data < outlier_threshold_low | column_data > outlier_threshold_high, na.rm = TRUE)
    percentage_outliers <- (outliers / total_values) * 100
  } else {
    outliers <- 0
    percentage_outliers <- NA  
  }
  
  # Determine variable type
  variable_type <- class(column_data)
  
  # Create result row
  result_row <- data.frame(
    Variable = column_name,
    Type = variable_type,
    NAs = na_count,
    Percentage_Outliers = percentage_outliers,
    Negative_Values = negative_count,
    Duplicate_Values = duplicate_count,
    Total_Values = total_values
  )
  
  return(result_row)
}

# Loop through each column and analyze
for (column in columns_to_analyze) {
  results <- rbind(results, analyze_column(column))
}

# Save results as a CSV file
write.csv(results, "gmr_data_analysis_results.csv", row.names = FALSE)

# Display results
print(results)



#DATA QUALITY TESTING FOR HEALTH DATASET
texas_health_data <- health_data %>%
  filter(state == "Texas")  

# Specify columns to analyze
columns_to_analyze <- c(
  "total_population",
  "area_sqmi",
  "population_density_per_sqmi",
  "fips",
  "percent_below_poverty",
  "percent_unemployed_CDC",
  "eightieth_percentile_income",  
  "twentieth_percentile_income",
  "income_ratio",
  "stay_at_home_announced",
  "stay_at_home_effective",
  "date_stay_at_home_announced",  
  "percent_fair_or_poor_health",
  "percent_smokers",
  "percent_adults_with_obesity",
  "percent_physically_inactive",
  "percent_with_access_to_exercise_opportunities",
  "percent_excessive_drinking",
  "percent_adults_with_diabetes",
  "percent_vaccinated",
  "lat",
  "lon"
)

# Initialize result data frame
results_health <- data.frame()

# Function to analyze each column
analyze_column <- function(column_name) {
  column_data <- texas_health_data[[column_name]]  
  
  # Calculate metrics
  total_values <- length(column_data)
  na_count <- sum(is.na(column_data))
  negative_count <- sum(column_data < 0, na.rm = TRUE)
  duplicate_count <- nrow(texas_health_data) - nrow(distinct(texas_health_data, !!sym(column_name)))  
  
  # Identify outliers using IQR method (only for numeric columns)
  if (is.numeric(column_data)) {
    Q1 <- quantile(column_data, 0.25, na.rm = TRUE)
    Q3 <- quantile(column_data, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    outlier_threshold_high <- Q3 + 1.5 * IQR
    outlier_threshold_low <- Q1 - 1.5 * IQR
    outliers <- sum(column_data < outlier_threshold_low | column_data > outlier_threshold_high, na.rm = TRUE)
    percentage_outliers <- (outliers / total_values) * 100
  } else {
    outliers <- 0
    percentage_outliers <- NA  # Not applicable for non-numeric columns
  }
  
  # Determine variable type
  variable_type <- class(column_data)
  
  # Create result row
  result_row <- data.frame(
    Variable = column_name,
    Type = variable_type,
    NAs = na_count,
    Percentage_Outliers = percentage_outliers,
    Negative_Values = negative_count,
    Duplicate_Values = duplicate_count,
    Total_Values = total_values
  )
  
  return(result_row)
}

# Loop through each column and analyze
for (column in columns_to_analyze) {
  results_health <- rbind(results_health, analyze_column(column))  # Corrected variable name
}

# Save results as a CSV file
write.csv(results_health, "health_data_analysis_results.csv", row.names = FALSE)

# Display results
print(results_health)



#OUTLIER DETECTION CORELATION GRAPH
tx_census_data <- census_data %>%
  filter(state == "TX")  

correlation_matrix <- cor(tx_census_data %>% select(total_pop, 
                                             commuters_by_public_transportation, 
                                             commuters_by_car_truck_van,
                                             commuters_by_subway_or_elevated,
                                             white_pop, black_pop, asian_pop, 
                                             hispanic_pop, amerindian_pop, 
                                             families_with_young_children,
                                             bachelors_degree, graduate_professional_degree, 
                                             bachelors_degree_or_higher_25_64), 
                          use = "complete.obs")

print(correlation_matrix)

#checking the outliers against population to see if they are truly outliers. 

tx_census_data <- census_data %>%
  filter(state == "TX") %>%
  select(total_pop, 
         commuters_by_public_transportation, 
         commuters_by_car_truck_van,
         commuters_by_subway_or_elevated,
         white_pop, 
         black_pop, 
         asian_pop, 
         hispanic_pop, 
         amerindian_pop, 
         families_with_young_children,
         bachelors_degree, 
         graduate_professional_degree, 
         bachelors_degree_or_higher_25_64)

# Calculate correlations with total_pop
correlations <- sapply(tx_census_data[-1], function(x) cor(tx_census_data$total_pop, x, use = "complete.obs"))

# Create a data frame for visualization
correlation_df <- data.frame(
  Variable = names(correlations),
  Correlation = correlations
)

# Plot the correlations with labels
ggplot(correlation_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = round(Correlation, 2)), vjust = -0.5, size = 4) +  # Add labels to the bars
  coord_flip() +  # Flip the coordinates for better readability
  labs(title = "Correlation of Variables with Total Population in Texas",
       x = "Variables",
       y = "Correlation Coefficient") +
  theme_minimal()



#OUTLIER DETECTION HISTORAM FOR SPREAD

# Filter for Texas and select relevant columns
tx_census_data <- census_data %>%
  filter(state == "TX") %>%
  select(total_pop, 
         commuters_by_public_transportation, 
         commuters_by_car_truck_van,
         commuters_by_subway_or_elevated,
         white_pop, 
         black_pop, 
         asian_pop, 
         hispanic_pop, 
         amerindian_pop, 
         families_with_young_children,
         bachelors_degree, 
         graduate_professional_degree, 
         bachelors_degree_or_higher_25_64)

# Create individual density plots
plot_list <- lapply(names(tx_census_data)[-1], function(var) {
  ggplot(tx_census_data, aes_string(x = var)) +
    geom_density(fill = "skyblue", alpha = 0.7) +
    labs(title = paste(var),
         x = var,
         y = "Density") +
    theme_minimal()
})

# Combine all plots in a grid
grid.arrange(grobs = plot_list, ncol = 3)  # Adjust ncol to change the number of columns in the grid


tx_census_data <- census_data %>%
  filter(state == "TX") %>%
  select(total_pop, 
         commuters_by_public_transportation, 
         commuters_by_car_truck_van,
         commuters_by_subway_or_elevated,
         white_pop, 
         black_pop, 
         asian_pop, 
         hispanic_pop, 
         amerindian_pop, 
         families_with_young_children,
         bachelors_degree, 
         graduate_professional_degree, 
         bachelors_degree_or_higher_25_64)

# Create individual histogram plots
plot_list <- lapply(names(tx_census_data)[-1], function(var) {
  ggplot(tx_census_data, aes_string(x = var)) +
    geom_histogram(binwidth = 1000, fill = "skyblue", color = "black", alpha = 0.7) +  # Adjust binwidth as needed
    labs(title = paste(var),
         x = var,
         y = "Frequency") +
    theme_minimal()
})

# Combine all plots in a grid
grid.arrange(grobs = plot_list, ncol = 3)



#SAVING THE FILTERED DATASETS AS SEPERATE FILES FOR LATER USE
# Filter the datasets
census_data_filtered <- census_data %>%
  filter(state == "TX") %>%
  select(
    total_pop, county_fips_code, commuters_by_public_transportation,
    commuters_by_car_truck_van, commuters_by_subway_or_elevated,
    white_pop, black_pop, asian_pop, hispanic_pop,
    amerindian_pop, other_race_pop, two_or_more_races_pop,
    not_hispanic_pop, median_income, median_age,
    families_with_young_children, high_school_diploma,
    bachelors_degree, graduate_professional_degree,
    bachelors_degree_or_higher_25_64, confirmed_cases, deaths
  )

gmr_data_filtered <- gmr_data %>%
  filter(sub_region_1 == "Texas") %>%
  select(
    sub_region_1, sub_region_2, metro_area,
    census_fips_code, transit_stations_percent_change_from_baseline,
    date, workplaces_percent_change_from_baseline
  )

health_data_filtered <- health_data %>%
  filter(state == "Texas") %>%
  select(
    total_population, area_sqmi, population_density_per_sqmi,
    fips, percent_below_poverty, percent_unemployed_CDC,
    eightieth_percentile_income, twentieth_percentile_income,
    income_ratio, stay_at_home_announced, stay_at_home_effective,
    date_stay_at_home_announced, percent_fair_or_poor_health,
    percent_smokers, percent_adults_with_obesity,
    percent_physically_inactive, percent_with_access_to_exercise_opportunities,
    percent_excessive_drinking, percent_adults_with_diabetes,
    percent_vaccinated, date, lat, lon
  )

# Save the filtered datasets as CSV files
write.csv(census_data_filtered, "census_data_filtered.csv", row.names = FALSE)
write.csv(gmr_data_filtered, "gmr_data_filtered.csv", row.names = FALSE)
write.csv(health_data_filtered, "health_data_filtered.csv", row.names = FALSE)



#GENERAL STATISTICS FOR EACH DATASET
# Function to calculate statistics
options(scipen = 999)  

# Function to calculate statistics and format the output as a data frame
calculate_statistics <- function(data) {
  stats <- data %>%
    summarise(
      across(where(is.numeric), list(
        mean = ~mean(.x, na.rm = TRUE),
        median = ~median(.x, na.rm = TRUE),
        variance = ~var(.x, na.rm = TRUE),
        range = ~paste(range(.x, na.rm = TRUE), collapse = " - "),  
        mode = ~as.numeric(names(sort(table(.x), decreasing = TRUE)[1]))
      ))
    )
  
  # Reshape the statistics to have variable names in the first column
  stats_long <- stats %>%
    pivot_longer(cols = everything(), 
                 names_to = c("Variable", ".value"), 
                 names_sep = "_") %>%
    rename(Range = range)  # Rename range column for clarity
  
  return(stats_long)
}


# Calculate statistics for each dataset
census_stats <- calculate_statistics(census_data_filtered)
gmr_stats <- calculate_statistics(gmr_data_filtered)
health_stats <- calculate_statistics(health_data_filtered)

# Print the statistics
print("Census Data Statistics:")
print(census_stats)

print("GMR Data Statistics:")
print(gmr_stats)

print("Health Data Statistics:")
print(health_stats)

# Save the statistics as CSV files
write.csv(census_stats, "census_data_statistics.csv", row.names = FALSE)
write.csv(gmr_stats, "gmr_data_statistics.csv", row.names = FALSE)
write.csv(health_stats, "health_data_statistics.csv", row.names = FALSE)
 


#VISUAL EXPLORATION
# Load the filtered datasets
census_data_filtered <- read.csv("census_data_filtered.csv")
gmr_data_filtered <- read.csv("gmr_data_filtered.csv")
health_data_filtered <- read.csv("health_data_filtered.csv")
# Check the structure of the loaded datasets
str(census_data_filtered)
str(gmr_data_filtered)
str(health_data_filtered)


options(scipen = 999)
# 1. Racial Distribution Stacked Bar Chart
# Get records with the least 10 populated counties
census_data_filtered <- read.csv("census_data_filtered.csv")

# Get records with the least 10 populated counties
least_counties <- census_data_filtered %>%
  arrange(total_pop) %>%
  slice(1:10)

# Get records with the top 10 populated counties
top_counties <- census_data_filtered %>%
  arrange(desc(total_pop)) %>%
  slice(1:10)

# Calculate racial composition as percentages for least populated counties
racial_least <- least_counties %>%
  summarise(
    white_pop = sum(white_pop, na.rm = TRUE),
    black_pop = sum(black_pop, na.rm = TRUE),
    asian_pop = sum(asian_pop, na.rm = TRUE),
    hispanic_pop = sum(hispanic_pop, na.rm = TRUE),
    amerindian_pop = sum(amerindian_pop, na.rm = TRUE),
    other_race_pop = sum(other_race_pop, na.rm = TRUE),
    two_or_more_races_pop = sum(two_or_more_races_pop, na.rm = TRUE)
  ) %>%
  mutate(total_racial_pop = white_pop + black_pop + asian_pop + hispanic_pop + amerindian_pop + other_race_pop + two_or_more_races_pop) %>%
  mutate(across(starts_with("white_pop"):starts_with("two_or_more_races_pop"), ~ .x / total_racial_pop * 100)) %>%  
  pivot_longer(cols = everything(), names_to = "Race", values_to = "Percentage") %>%
  mutate(Type = "Least Populated")

# Calculate racial composition as percentages for top populated counties
racial_top <- top_counties %>%
  summarise(
    white_pop = sum(white_pop, na.rm = TRUE),
    black_pop = sum(black_pop, na.rm = TRUE),
    asian_pop = sum(asian_pop, na.rm = TRUE),
    hispanic_pop = sum(hispanic_pop, na.rm = TRUE),
    amerindian_pop = sum(amerindian_pop, na.rm = TRUE),
    other_race_pop = sum(other_race_pop, na.rm = TRUE),
    two_or_more_races_pop = sum(two_or_more_races_pop, na.rm = TRUE)
  ) %>%
  mutate(total_racial_pop = white_pop + black_pop + asian_pop + hispanic_pop + amerindian_pop + other_race_pop + two_or_more_races_pop) %>%
  mutate(across(starts_with("white_pop"):starts_with("two_or_more_races_pop"), ~ .x / total_racial_pop * 100)) %>%  # Convert to percentages
  pivot_longer(cols = everything(), names_to = "Race", values_to = "Percentage") %>%
  mutate(Type = "Top Populated")

# Combine both datasets
racial_combined <- bind_rows(racial_least, racial_top)

# Create a stacked bar chart with equal lengths and percentage labels
ggplot(racial_combined, aes(x = Type, y = Percentage, fill = Race)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")), 
            position = position_stack(vjust = 0.5),  # Place labels in the middle of each segment
            color = "white") +  # Label color
  labs(title = "Racial Composition of Top and Least Populated Counties", 
       x = "County Population Type", 
       y = "Percentage", 
       fill = "Race") +
  scale_y_continuous(labels = scales::percent_format(scale = 1), limits = c(0, 100)) +  # Ensure y-axis goes to 100%
  theme_minimal()


# 2. Median Income Histogram
# Summarize median income data for least populated counties
median_income_least <- least_counties %>%
  summarise(
    median_income = sum(median_income, na.rm = TRUE),
    total_counties = n()
  ) %>%
  mutate(average_income = median_income / total_counties) %>%
  mutate(Type = "Least Populated")

# Summarize median income data for top populated counties
median_income_top <- top_counties %>%
  summarise(
    median_income = sum(median_income, na.rm = TRUE),
    total_counties = n()
  ) %>%
  mutate(average_income = median_income / total_counties) %>%
  mutate(Type = "Top Populated")

# Combine both datasets
median_income_combined <- bind_rows(
  median_income_least %>% select(Type, average_income),
  median_income_top %>% select(Type, average_income)
)

# Prepare data for stacked bar chart
median_income_long <- median_income_combined %>%
  pivot_longer(cols = average_income, names_to = "Metric", values_to = "Value")

# Create a stacked bar chart for median income
ggplot(median_income_long, aes(x = Type, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Value, 1))), 
            position = position_stack(vjust = 0.5),  # Place labels in the middle of each segment
            color = "white") +  # Label color
  labs(title = "Average Median Income of Top and Least Populated Counties", 
       x = "County Population Type", 
       y = "Average Median Income", 
       fill = "Metric") +
  scale_y_continuous(labels = scales::dollar_format(scale = 1)) +  # Format y-axis as currency
  theme_minimal()

# 3. Percent Below Poverty Boxplot
# Summarize transportation data for least populated counties
transport_least <- least_counties %>%
  summarise(
    commuters_by_public_transportation = sum(commuters_by_public_transportation, na.rm = TRUE),
    commuters_by_car_truck_van = sum(commuters_by_car_truck_van, na.rm = TRUE),
    commuters_by_subway_or_elevated = sum(commuters_by_subway_or_elevated, na.rm = TRUE)
  ) %>%
  mutate(total_commuters = commuters_by_public_transportation + commuters_by_car_truck_van + commuters_by_subway_or_elevated) %>%
  mutate(across(starts_with("commuters"), ~ .x / total_commuters * 100)) %>%  # Convert to percentages
  pivot_longer(cols = everything(), names_to = "TransportMode", values_to = "Percentage") %>%
  mutate(Type = "Least Populated")

# Summarize transportation data for top populated counties
transport_top <- top_counties %>%
  summarise(
    commuters_by_public_transportation = sum(commuters_by_public_transportation, na.rm = TRUE),
    commuters_by_car_truck_van = sum(commuters_by_car_truck_van, na.rm = TRUE),
    commuters_by_subway_or_elevated = sum(commuters_by_subway_or_elevated, na.rm = TRUE)
  ) %>%
  mutate(total_commuters = commuters_by_public_transportation + commuters_by_car_truck_van + commuters_by_subway_or_elevated) %>%
  mutate(across(starts_with("commuters"), ~ .x / total_commuters * 100)) %>%  # Convert to percentages
  pivot_longer(cols = everything(), names_to = "TransportMode", values_to = "Percentage") %>%
  mutate(Type = "Top Populated")

# Combine both datasets
transport_combined <- bind_rows(transport_least, transport_top)

# Create a stacked bar chart for transportation percentages
ggplot(transport_combined, aes(x = Type, y = Percentage, fill = TransportMode)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")), 
            position = position_stack(vjust = 0.5),  # Place labels in the middle of each segment
            color = "white") +  # Label color
  labs(title = "Transportation Makeup of Top and Least Populated Counties", 
       x = "County Population Type", 
       y = "Percentage", 
       fill = "Transportation Mode") +
  scale_y_continuous(labels = scales::percent_format(scale = 1), limits = c(0, 100)) +  # Ensure y-axis goes to 100%
  theme_minimal()


#COMBINING DATASETS

health_data_filtered$date <- as.Date(health_data_filtered$date, format = "%Y-%m-%d")

# Select the first record for each fips code (earliest date)
health_data_first <- health_data_filtered %>%
  group_by(fips) %>%
  slice(which.min(date)) %>%  # Get the record with the earliest date
  ungroup()

# Merge the two datasets on county_fips_code and fips
combined_data <- census_data_filtered %>%
  left_join(health_data_first, by = c("county_fips_code" = "fips"))

# Check the structure of the combined dataset
str(combined_data)

# Check for mismatched records
mismatched_records <- combined_data %>%
  filter(is.na(county_fips_code))  # Check for records in census data that did not find a match

# Display mismatched records
print("Mismatched Records:")
print(mismatched_records)

str(combined_data)
head(combined_data)
write.csv(combined_data, "combined_data.csv", row.names = FALSE)


#HEAT MAPS
# Load the GIS data
gis_data <- read.csv("Texas_Counties_GIS_Data_Final.csv")
combined_data <- read.csv("combined_data.csv")

# Remove commas from FIPS in gis_data
gis_data <- gis_data %>%
  mutate(FIPS = gsub(",", "", FIPS))  

# Convert county_fips_code in combined_data to character, ensuring proper format
combined_data <- combined_data %>%
  mutate(FIPS = sprintf("%05d", county_fips_code))  

# FIPS column in gis_data needs to be a character
gis_data <- gis_data %>%
  mutate(FIPS = as.character(FIPS))  

# Merge the GIS data with the combined data
merged_data <- gis_data %>%
  left_join(combined_data, by = "FIPS")

# Check if the merge was successful
head(merged_data)

# Create a heat map for Median Income using geom_point
ggplot(merged_data, aes(x = Y_Longitude, y = X_Latitude)) + 
  geom_point(aes(color = median_income), alpha = 0.7, size = 3) +  # Use points for individual locations
  scale_color_gradient(low = "yellow", high = "red") +
  coord_quickmap() + 
  labs(title = "Median Income Heat Map for Texas", 
       fill = "Median Income") +
  theme_minimal() +
  theme(legend.position = "right")



ggplot(merged_data, aes(x = Y_Longitude, y = X_Latitude)) + 
  geom_point(aes(color = stay_at_home_effective), alpha = 0.7, size = 3) +
  scale_color_manual(values = c("yes" = "green", "no" = "red")) +
  coord_quickmap() + 
  labs(title = "Stay-at-Home Order Status in Texas", 
       color = "Stay-at-Home Order") +
  theme_minimal() +
  theme(legend.position = "right")


ggplot(merged_data, aes(x = Y_Longitude, y = X_Latitude)) + 
  geom_point(aes(color = median_age), alpha = 0.7, size = 3) + 
  scale_color_gradient(low = "blue", high = "red") +
  coord_quickmap() + 
  labs(title = "Median Age Heat Map for Texas", 
       fill = "Median Age") +
  theme_minimal() +
  theme(legend.position = "right")

# Heat map for Percentage of Smokers
ggplot(merged_data, aes(x = Y_Longitude, y = X_Latitude)) + 
  geom_point(aes(color = percent_smokers), alpha = 0.7, size = 3) + 
  scale_color_gradient(low = "yellow", high = "red") +
  coord_quickmap() + 
  labs(title = "Percentage of Smokers Heat Map for Texas", 
       fill = "Percentage of Smokers") +
  theme_minimal() +
  theme(legend.position = "right")

# Heat map for Percentage of Fair or Poor Health
ggplot(merged_data, aes(x = Y_Longitude, y = X_Latitude)) + 
  geom_point(aes(color = percent_fair_or_poor_health), alpha = 0.7, size = 3) + 
  scale_color_gradient(low = "lightblue", high = "darkblue") +
  coord_quickmap() + 
  labs(title = "Percentage of Fair or Poor Health Heat Map for Texas", 
       fill = "Percentage of Fair or Poor Health") +
  theme_minimal() +
  theme(legend.position = "right")


##CORELATION TESTING
# Correlation Analysis
combined_data <- read.csv("combined_data.csv")
head(combined_data)

# Convert the 'stay_at_home_effective' variable to numeric
correlation_data <- combined_data %>%
  mutate(stay_at_home_effective_numeric = ifelse(stay_at_home_effective == "yes", 1, 0))  # Convert to numeric

#convert relevant data to per 1000 so that we have a better understanding when comparing
correlation_data <- correlation_data %>%
  mutate(
    confirmed_cases_per_thousand = confirmed_cases / (total_population / 1000),  # Cases per 1000 people
    deaths_per_thousand = deaths / (total_population / 1000),  # Deaths per 1000 people
    commuters_by_public_transportation_per_thousand = commuters_by_public_transportation / (total_population / 1000),  # Per 1000
    commuters_by_car_truck_van_per_thousand = commuters_by_car_truck_van / (total_population / 1000),  # Per 1000
    white_pop_per_thousand = white_pop / (total_population / 1000),  # Per 1000
    black_pop_per_thousand = black_pop / (total_population / 1000),  # Per 1000
    hispanic_pop_per_thousand = hispanic_pop / (total_population / 1000)  # Per 1000
  )

print(colnames(correlation_data))

# Select relevant columns for correlation
correlation_cases <- cor(correlation_data %>%
                           select(
                             confirmed_cases_per_thousand, 
                             deaths_per_thousand,
                             population_density_per_sqmi,
                             commuters_by_public_transportation_per_thousand,
                             commuters_by_car_truck_van_per_thousand,
                             white_pop_per_thousand,
                             black_pop_per_thousand,
                             hispanic_pop_per_thousand,
                             median_income,
                             stay_at_home_effective_numeric,  # Use the numeric version
                             median_age,
                             families_with_young_children,
                             high_school_diploma,
                             bachelors_degree,
                             percent_vaccinated,
                             percent_fair_or_poor_health,
                             percent_smokers,
                             percent_adults_with_obesity,
                             percent_physically_inactive,
                             percent_with_access_to_exercise_opportunities,
                             percent_excessive_drinking,
                             percent_adults_with_diabetes
                           ), 
                         use = "complete.obs")

# Visualize the correlation matrix for confirmed cases
corrplot(correlation_cases, method = "circle", title = "Correlation Matrix for Confirmed Cases (per Thousand)")

correlation_cases_df <- as.data.frame(correlation_cases)
print(correlation_cases_df)

# For total deaths
correlation_deaths <- cor(correlation_data %>%
                            select(
                              deaths_per_thousand,
                              confirmed_cases_per_thousand,
                              population_density_per_sqmi,
                              commuters_by_public_transportation_per_thousand,
                              commuters_by_car_truck_van_per_thousand,
                              white_pop_per_thousand,
                              black_pop_per_thousand,
                              hispanic_pop_per_thousand,
                              median_income,
                              stay_at_home_effective_numeric,  
                              median_age,
                              families_with_young_children,
                              high_school_diploma,
                              bachelors_degree,
                              percent_vaccinated,
                              percent_fair_or_poor_health,
                              percent_smokers,
                              percent_adults_with_obesity,
                              percent_physically_inactive,
                              percent_with_access_to_exercise_opportunities,
                              percent_excessive_drinking,
                              percent_adults_with_diabetes
                            ), 
                          use = "complete.obs")

# Visualize the correlation matrix for total deaths
corrplot(correlation_deaths, method = "circle", title = "Correlation Matrix for Total Deaths (per Thousand)")
correlation_deaths_df <- as.data.frame(correlation_deaths)
print(correlation_deaths_df)
