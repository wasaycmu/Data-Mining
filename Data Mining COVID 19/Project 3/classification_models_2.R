# Load required libraries
library(tidyverse)      # For data manipulation and visualization
library(caret)          # For machine learning functions
library(randomForest)   # For Random Forest model
library(xgboost)        # For XGBoost model (graduate requirement)
library(e1071)          # For SVM model
library(ROSE)           # For handling imbalanced data
library(corrplot)       # For correlation visualization
library(pROC)          # For ROC curves
library(gridExtra)      # For arranging multiple plots
library(viridis)        # For color schemes
library(MLmetrics)

# Read the data
covid_data <- read.csv("combined_data.csv")

#--------------------
# 1. Data Preparation
#--------------------

# Create risk classes based on deaths per 1000 population
covid_data <- covid_data %>%
  mutate(deaths_per_1000 = (deaths / total_population) * 1000,
         risk_class = case_when(
           deaths_per_1000 <= quantile(deaths_per_1000, 0.33) ~ "low",
           deaths_per_1000 <= quantile(deaths_per_1000, 0.66) ~ "medium",
           TRUE ~ "high"
         ))

# Convert risk_class to factor
covid_data$risk_class <- as.factor(covid_data$risk_class)

# Select predictive features
selected_features <- c(
  "median_income", "median_age", "percent_below_poverty",
  "percent_unemployed_CDC", "percent_fair_or_poor_health",
  "percent_smokers", "percent_adults_with_obesity",
  "percent_physically_inactive", "percent_with_access_to_exercise_opportunities",
  "percent_excessive_drinking", "percent_adults_with_diabetes",
  "percent_vaccinated", "population_density_per_sqmi",
  "bachelors_degree_or_higher_25_64", "risk_class"
)

# Create modeling dataset
model_data <- covid_data[, selected_features]

# Handle missing values
model_data <- model_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Create derived features
model_data <- model_data %>%
  mutate(
    health_risk_score = (percent_fair_or_poor_health + percent_smokers + 
                           percent_adults_with_obesity) / 3,
    socioeconomic_score = (percent_below_poverty + percent_unemployed_CDC) / 2
  )


# Calculate statistics for the derived features
derived_features_stats <- model_data %>%
  summarise(
    health_risk_score_mean = mean(health_risk_score, na.rm = TRUE),
    health_risk_score_median = median(health_risk_score, na.rm = TRUE),
    health_risk_score_sd = sd(health_risk_score, na.rm = TRUE),
    health_risk_score_min = min(health_risk_score, na.rm = TRUE),
    health_risk_score_max = max(health_risk_score, na.rm = TRUE),
    health_risk_score_variance = var(health_risk_score, na.rm = TRUE),
    
    socioeconomic_score_mean = mean(socioeconomic_score, na.rm = TRUE),
    socioeconomic_score_median = median(socioeconomic_score, na.rm = TRUE),
    socioeconomic_score_sd = sd(socioeconomic_score, na.rm = TRUE),
    socioeconomic_score_min = min(socioeconomic_score, na.rm = TRUE),
    socioeconomic_score_max = max(socioeconomic_score, na.rm = TRUE),
    socioeconomic_score_variance = var(socioeconomic_score, na.rm = TRUE)
  )

# Print the statistics
print(derived_features_stats)

# Calculate statistics for the risk_class
risk_class_stats <- model_data %>%
  group_by(risk_class) %>%
  summarise(
    Count = n(),                               # Total count of each risk class
    Proportion = n() / nrow(model_data) * 100  # Proportion as a percentage
  )

# Print the statistics
print(risk_class_stats)

#--------------------
# 2. Model Training
#--------------------
# Initialize a list to store times
model_times <- list()

# Split data into training and testing sets
set.seed(111)
trainIndex <- createDataPartition(model_data$risk_class, p = 0.8, list = FALSE)
training <- model_data[trainIndex,]
testing <- model_data[-trainIndex,]

# Create cross-validation control
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = TRUE
)

# Train Random Forest model and record time
set.seed(111)
model_times$RandomForest <- system.time({
  rf_model <- train(
    risk_class ~ .,
    data = training,
    method = "rf",
    trControl = ctrl,
    metric = "Accuracy",
    importance = TRUE
  )
})

# Train SVM model and record time
set.seed(111)
model_times$SVM <- system.time({
  svm_model <- train(
    risk_class ~ .,
    data = training,
    method = "svmRadial",
    trControl = ctrl,
    metric = "Accuracy"
  )
})

# Train XGBoost model and record time
set.seed(111)
model_times$XGBoost <- suppressWarnings(system.time({
  xgb_model <- train(
    risk_class ~ .,
    data = training,
    method = "xgbTree",
    trControl = ctrl,
    metric = "Accuracy"
  )
}))

# Train Neural Network model and record time
set.seed(111)
model_times$NeuralNetwork <- system.time({
  nnet_model <- train(
    risk_class ~ .,
    data = training,
    method = "nnet",
    trControl = ctrl,
    metric = "Accuracy",
    trace = FALSE
  )
})

# Train Logistic Regression model and record time
set.seed(111)
model_times$LogisticRegression <- system.time({
  logistic_model <- train(
    risk_class ~ .,
    data = training,
    method = "multinom",
    trControl = ctrl,
    metric = "Accuracy"
  )
})



#--------------------
# 3. Model Evaluation
#--------------------

# Create predictions
predictions_logistic <- predict(logistic_model, testing)
predictions_rf <- predict(rf_model, testing)
predictions_svm <- predict(svm_model, testing)
predictions_xgb <- predict(xgb_model, testing)
predictions_nnet <- predict(nnet_model, testing)

# Calculate performance metrics for each model
evaluate_model <- function(predictions, actual) {
  cm <- confusionMatrix(predictions, actual)
  return(list(
    accuracy = cm$overall["Accuracy"],
    kappa = cm$overall["Kappa"],
    sensitivity = mean(cm$byClass[, "Sensitivity"]),
    specificity = mean(cm$byClass[, "Specificity"])
  ))
}

# Get evaluation metrics
logistic_eval <- evaluate_model(predictions_logistic, testing$risk_class)
rf_eval <- evaluate_model(predictions_rf, testing$risk_class)
svm_eval <- evaluate_model(predictions_svm, testing$risk_class)
xgb_eval <- evaluate_model(predictions_xgb, testing$risk_class)
nnet_eval <- evaluate_model(predictions_nnet, testing$risk_class)

# Create model comparison dataframe
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM", "XGBoost", "Neural Network"),
  Accuracy = c(logistic_eval$accuracy, rf_eval$accuracy, svm_eval$accuracy, xgb_eval$accuracy, nnet_eval$accuracy),
  Kappa = c(logistic_eval$kappa, rf_eval$kappa, svm_eval$kappa, xgb_eval$kappa, nnet_eval$kappa),
  Sensitivity = c(logistic_eval$sensitivity, rf_eval$sensitivity, svm_eval$sensitivity, xgb_eval$sensitivity, nnet_eval$sensitivity),
  Specificity = c(logistic_eval$specificity, rf_eval$specificity, svm_eval$specificity, xgb_eval$specificity, nnet_eval$specificity)
)

#best parameters for the modeling
best_params <- list(
  RandomForest = rf_model$bestTune,
  SVM = svm_model$bestTune,
  XGBoost = xgb_model$bestTune,
  NeuralNetwork = nnet_model$bestTune,
  LogisticRegression = logistic_model$bestTune
)

# Display the best hyperparameters
print(best_params)

#--------------------
# 4. Visualizations
#--------------------
# Display model times
model_times


#individual model performnace metrics:
# Function to evaluate a model and output metrics
evaluate_and_report <- function(model_name, predictions, actual) {
  cat("\n==============================\n")
  cat("Performance Metrics for:", model_name, "\n")
  cat("==============================\n")
  
  # Generate confusion matrix
  cm <- confusionMatrix(predictions, actual)
  
  # Extract metrics
  accuracy <- cm$overall["Accuracy"]
  kappa <- cm$overall["Kappa"]
  sensitivity <- mean(cm$byClass[, "Sensitivity"], na.rm = TRUE)
  specificity <- mean(cm$byClass[, "Specificity"], na.rm = TRUE)
  precision <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
  f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
  
  # Print metrics
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("Kappa:", round(kappa, 3), "\n")
  cat("Sensitivity:", round(sensitivity, 3), "\n")
  cat("Specificity:", round(specificity, 3), "\n")
  cat("Precision:", round(precision, 3), "\n")
  cat("F1-Score:", round(f1, 3), "\n")
}

# Evaluate each model
evaluate_and_report("Random Forest", predictions_rf, testing$risk_class)
evaluate_and_report("SVM", predictions_svm, testing$risk_class)
evaluate_and_report("XGBoost", predictions_xgb, testing$risk_class)
evaluate_and_report("Neural Network", predictions_nnet, testing$risk_class)
evaluate_and_report("Logistic Regression", predictions_logistic, testing$risk_class)



# Model performance comparison plot
model_comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Performance Comparison (Accuracy)",
       y = "Accuracy",
       x = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_viridis_d() +
  geom_text(aes(label = sprintf("%.1f%%", Accuracy * 100)), vjust = -0.5)

# Feature importance plot (for Random Forest)
importance_df <- varImp(rf_model)$importance %>%
  as.data.frame() %>%
  rownames_to_column("Feature")

importance_plot <- ggplot(importance_df %>% 
                            arrange(desc(high)) %>% 
                            head(10),
                          aes(x = reorder(Feature, high), y = high)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Most Important Features",
       x = "",
       y = "Importance Score")

# Create box plots for key features
key_features <- c("median_income", "percent_vaccinated", "percent_fair_or_poor_health")
plot_list <- list()

for(feature in key_features) {
  p <- ggplot(model_data, aes_string(x = "risk_class", y = feature, fill = "risk_class")) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = paste("Distribution of", feature, "by Risk Class"),
         x = "Risk Class",
         y = feature) +
    scale_fill_viridis_d() +
    theme(legend.position = "none")
  plot_list[[feature]] <- p
}

# Arrange plots
grid.arrange(
  model_comparison_plot,
  importance_plot,
  ncol = 2
)

do.call(grid.arrange, c(plot_list, ncol = 2))

#--------------------
# 5. Save Results
#--------------------

# Save plots
ggsave("model_performance.png", model_comparison_plot, width = 10, height = 6)
ggsave("feature_importance.png", importance_plot, width = 10, height = 6)
ggsave("feature_distributions.png", marrangeGrob(plot_list, nrow=2, ncol=2), width = 12, height = 8)

# Save detailed results to file

sink("model_analysis_results.txt")
cat("COVID-19 Risk Classification Analysis Results\n\n")
cat("1. Model Performance Comparison:\n")
print(model_comparison)
cat("\n\n2. Random Forest Feature Importance:\n")
print(varImp(rf_model))
cat("\n\n3. Confusion Matrices:\n")
cat("\nRandom Forest:\n")
print(confusionMatrix(predictions_rf, testing$risk_class))
cat("\nSVM:\n")
print(confusionMatrix(predictions_svm, testing$risk_class))
cat("\nXGBoost:\n")
print(confusionMatrix(predictions_xgb, testing$risk_class))
cat("\nNeural Network:\n")
print(confusionMatrix(predictions_nnet, testing$risk_class))
sink("model_analysis_results.txt", append = TRUE)
cat("\nLogistic Regression:\n")
print(confusionMatrix(predictions_logistic, testing$risk_class))
sink()

# Print summary results to console
print("Model Comparison:")
print(model_comparison)
print("\nFeature Importance (Top 10):")
print(importance_df %>% arrange(desc(high)) %>% head(10))

# Kappa comparison plot
kappa_comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Kappa, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Performance Comparison (Kappa)",
       y = "Kappa",
       x = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_viridis_d() +
  geom_text(aes(label = sprintf("%.3f", Kappa)), vjust = -0.5)

ggsave("model_performance_kappa.png", kappa_comparison_plot, width = 10, height = 6)


# Sensitivity comparison plot
sensitivity_comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Sensitivity, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Performance Comparison (Sensitivity)",
       y = "Sensitivity",
       x = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_viridis_d() +
  geom_text(aes(label = sprintf("%.3f", Sensitivity)), vjust = -0.5)

# Save the plot
ggsave("model_performance_sensitivity.png", sensitivity_comparison_plot, width = 10, height = 6)


# Specificity comparison plot
specificity_comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Specificity, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Performance Comparison (Specificity)",
       y = "Specificity",
       x = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_viridis_d() +
  geom_text(aes(label = sprintf("%.3f", Specificity)), vjust = -0.5)

# Save the plot
ggsave("model_performance_specificity.png", specificity_comparison_plot, width = 10, height = 6)


# Melt data to long format for combined plotting
library(reshape2)
combined_metrics <- model_comparison %>%
  select(Model, Accuracy, Kappa, Sensitivity, Specificity) %>%
  melt(id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Create a combined bar plot with metrics on x-axis and models differentiated by color
combined_metrics_plot <- ggplot(combined_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Performance Comparison by Metric",
       y = "Metric Value",
       x = "Metric") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_viridis_d(name = "Model") +
  geom_text(aes(label = sprintf("%.3f", Value)), position = position_dodge(width = 0.9), vjust = -0.5)

# Save the plot
ggsave("model_performance_by_metric.png", combined_metrics_plot, width = 12, height = 6)




library(pROC)

evaluate_and_report <- function(model_name, predictions, actual, predicted_probs = NULL, model = NULL) {
  cat("\n==============================\n")
  cat("Performance Metrics for:", model_name, "\n")
  cat("==============================\n")
  
  # Generate confusion matrix
  cm <- confusionMatrix(predictions, actual)
  
  # Extract metrics
  accuracy <- cm$overall["Accuracy"]
  kappa <- cm$overall["Kappa"]
  sensitivity <- mean(cm$byClass[, "Sensitivity"], na.rm = TRUE)
  specificity <- mean(cm$byClass[, "Specificity"], na.rm = TRUE)
  precision <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
  f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
  
  # Calculate AUC-ROC if predicted probabilities are available
  auc_roc <- NA
  if (!is.null(predicted_probs)) {
    roc_obj <- roc(actual, predicted_probs[, 2])
    auc_roc <- auc(roc_obj)
    cat("AUC-ROC:", round(auc_roc, 3), "\n")
  }
  
  # Print metrics
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("Kappa:", round(kappa, 3), "\n")
  cat("Sensitivity:", round(sensitivity, 3), "\n")
  cat("Specificity:", round(specificity, 3), "\n")
  cat("Precision:", round(precision, 3), "\n")
  cat("F1-Score:", round(f1, 3), "\n")
  
  # Include feature importance for XGBoost
  if (model_name == "XGBoost" && !is.null(model)) {
    cat("\nFeature Importance for XGBoost:\n")
    xgb_importance <- xgb.importance(feature_names = colnames(training[, -ncol(training)]), 
                                     model = model$finalModel)
    print(xgb_importance)
  }
  
  # Save AUC-ROC to file
  if (!is.na(auc_roc)) {
    auc_file <- paste0(model_name, "_AUC_ROC.txt")
    write(paste0("AUC-ROC for ", model_name, ": ", round(auc_roc, 3)), file = auc_file)
    cat("AUC-ROC saved to", auc_file, "\n")
  }
}

# Evaluate each model
evaluate_and_report("Random Forest", predictions_rf, testing$risk_class, predicted_probs_rf)
evaluate_and_report("SVM", predictions_svm, testing$risk_class, predicted_probs_svm)
evaluate_and_report("XGBoost", predictions_xgb, testing$risk_class, predicted_probs_xgb, xgb_model)
evaluate_and_report("Neural Network", predictions_nnet, testing$risk_class, predicted_probs_nnet)
evaluate_and_report("Logistic Regression", predictions_logistic, testing$risk_class, predicted_probs_logistic)

# Save detailed results to file
sink("model_analysis_results.txt")
cat("COVID-19 Risk Classification Analysis Results\n\n")
cat("1. Model Performance Comparison:\n")
print(model_comparison)
cat("\n\n2. Random Forest Feature Importance:\n")
print(varImp(rf_model))
cat("\n\n3. Confusion Matrices:\n")
cat("\nRandom Forest:\n")
print(confusionMatrix(predictions_rf, testing$risk_class))
cat("\nSVM:\n")
print(confusionMatrix(predictions_svm, testing$risk_class))
cat("\nXGBoost:\n")
print(confusionMatrix(predictions_xgb, testing$risk_class))
cat("\nXGBoost Feature Importance:\n")
if (!is.null(xgb_importance)) print(xgb_importance)
cat("\nNeural Network:\n")
print(confusionMatrix(predictions_nnet, testing$risk_class))
sink("model_analysis_results.txt", append = TRUE)
cat("\nLogistic Regression:\n")
print(confusionMatrix(predictions_logistic, testing$risk_class))
sink()


#--------------------
# 6. Understanding Performance Results
#--------------------


# Get all feature names used for training (excluding the target variable 'risk_class')
feature_names <- colnames(model_data)[colnames(model_data) != "risk_class"]

# Initialize a list to store descriptive statistics
stats_list <- list()

# Create box plots for all features and calculate statistics
for (feature in feature_names) {
  
  # Create box plot
  p <- ggplot(model_data, aes_string(x = "risk_class", y = feature, fill = "risk_class")) +
    geom_boxplot(outlier.size = 0.5) +
    theme_minimal() +
    labs(title = paste("Distribution of", feature, "by Risk Class"),
         x = "Risk Class",
         y = feature) +
    scale_fill_viridis_d() +
    theme(legend.position = "none",
          plot.title = element_text(size = 10, hjust = 0.5))
  
  # Save each plot as an image
  plot_filename <- paste0("boxplot_", feature, ".png")
  ggsave(plot_filename, p, width = 6, height = 4)
  
  # Calculate descriptive statistics for the feature grouped by risk_class
  stats <- model_data %>%
    group_by(risk_class) %>%
    summarize(
      Mean = mean(.data[[feature]], na.rm = TRUE),
      Median = median(.data[[feature]], na.rm = TRUE),
      Std_Dev = sd(.data[[feature]], na.rm = TRUE),
      Min = min(.data[[feature]], na.rm = TRUE),
      Max = max(.data[[feature]], na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    mutate(Feature = feature) %>%
    relocate(Feature, .before = risk_class)
  
  # Add to the list of statistics
  stats_list[[feature]] <- stats
}

# Combine all statistics into a single data frame
stats_combined <- bind_rows(stats_list)

# Save the statistics to a CSV file
write.csv(stats_combined, "feature_statistics_by_risk_class.csv", row.names = FALSE)

cat("Box plots saved in the 'plots' folder.\n")
cat("Descriptive statistics saved to 'feature_statistics_by_risk_class.csv'.\n")

