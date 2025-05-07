# Required libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(NbClust)
library(mclust)
library(gridExtra)
library(scales)
library(clValid)
library(dendextend)
library(dbscan)
library(poLCA)
library(fpc)

# Load and prepare data
data <- read.csv("combined_data.csv")

# First calculate the ratios and rates
data <- data %>%
  dplyr::mutate(
    hispanic_ratio = hispanic_pop / total_pop,
    white_ratio = white_pop / total_pop,
    black_ratio = black_pop / total_pop,
    cases_per_1000 = (confirmed_cases / total_pop) * 1000,
    deaths_per_1000 = (deaths / total_pop) * 1000
  )

# Then select features for clustering using dplyr::select
features_for_clustering <- data %>%
  dplyr::select(
    population_density_per_sqmi,
    hispanic_ratio,
    white_ratio,
    black_ratio,
    percent_fair_or_poor_health,
    percent_adults_with_obesity,
    percent_smokers,
    percent_vaccinated,
    median_income,
    percent_below_poverty,
    cases_per_1000,
    deaths_per_1000
  ) %>%
  na.omit()

# Scale the features
scaled_features <- scale(as.matrix(features_for_clustering))
rownames(scaled_features) <- rownames(features_for_clustering)

# Keep track of counties after na.omit
valid_counties_indices <- which(complete.cases(features_for_clustering))
valid_counties <- data[valid_counties_indices, ]

# Find optimal number of clusters using elbow method
wss <- numeric(10)
for(k in 1:10) {
  km <- kmeans(scaled_features, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}

# Plot elbow curve
plot(1:10, wss, type = "b", 
     xlab = "Number of Clusters (k)", 
     ylab = "Total Within Sum of Squares",
     main = "Elbow Method for Optimal k")

# 1. K-means Clustering
kmeans_result <- kmeans(scaled_features, centers = 5, nstart = 25)

# 2. Hierarchical Clustering
dist_matrix <- dist(scaled_features, method = "euclidean")
hc_ward <- hclust(dist_matrix, method = "ward.D2")
hc_clusters <- cutree(hc_ward, k = 5)

# 3. DBSCAN Clustering
# Determine eps using k-nearest neighbor distance
kNNdist <- dbscan::kNNdist(scaled_features, k = 4)
eps <- mean(kNNdist)
dbscan_result <- dbscan::dbscan(scaled_features, eps = eps, minPts = 4)

# 4. Model-based Clustering
mclust_result <- Mclust(scaled_features, G = 1:10)
mclust_clusters <- mclust_result$classification

# Create visualizations
p1 <- fviz_cluster(list(data = scaled_features, cluster = kmeans_result$cluster),
                   main = "K-means Clustering")

p2 <- fviz_dend(hc_ward, k = 5, main = "Hierarchical Clustering")

p3 <- fviz_cluster(list(data = scaled_features, cluster = dbscan_result$cluster),
                   main = "DBSCAN Clustering")

p4 <- fviz_cluster(list(data = scaled_features, cluster = mclust_clusters),
                   main = "Model-based Clustering")

# Arrange plots
grid.arrange(p1, p2, p3, p4, ncol = 2)

# Create cluster summaries
create_cluster_summary <- function(clusters, data) {
  summary_df <- data.frame(
    cluster = clusters,
    stringsAsFactors = FALSE
  )
  summary_df <- cbind(summary_df, data)
  
  summary_stats <- aggregate(. ~ cluster, data = summary_df[, c("cluster", 
                                                                "total_pop", 
                                                                "cases_per_1000", 
                                                                "deaths_per_1000", 
                                                                "percent_vaccinated", 
                                                                "median_income", 
                                                                "percent_below_poverty")], 
                             FUN = mean)
  
  # Add count of counties per cluster
  cluster_counts <- table(clusters)
  summary_stats$n_counties <- cluster_counts[match(summary_stats$cluster, names(cluster_counts))]
  
  return(summary_stats)
}

# Calculate summaries
kmeans_summary <- create_cluster_summary(kmeans_result$cluster, valid_counties)
hc_summary <- create_cluster_summary(hc_clusters, valid_counties)
dbscan_summary <- create_cluster_summary(dbscan_result$cluster, valid_counties)
mclust_summary <- create_cluster_summary(mclust_clusters, valid_counties)

# Print summaries
print("K-means Clustering Summary:")
print(kmeans_summary)
print("\nHierarchical Clustering Summary:")
print(hc_summary)
print("\nDBSCAN Clustering Summary:")
print(dbscan_summary)
print("\nModel-based Clustering Summary:")
print(mclust_summary)

# Calculate silhouette scores
silhouette_kmeans <- mean(silhouette(kmeans_result$cluster, dist_matrix)[,3])
silhouette_hc <- mean(silhouette(hc_clusters, dist_matrix)[,3])
silhouette_dbscan <- mean(silhouette(dbscan_result$cluster, dist_matrix)[,3])
silhouette_mclust <- mean(silhouette(mclust_clusters, dist_matrix)[,3])

# Create evaluation metrics dataframe
eval_metrics <- data.frame(
  Method = c("K-means", "Hierarchical", "DBSCAN", "Model-based"),
  Silhouette = c(silhouette_kmeans, silhouette_hc, silhouette_dbscan, silhouette_mclust)
)

# Print evaluation metrics
print("\nClustering Evaluation Metrics:")
print(eval_metrics)

# Save results
write.csv(kmeans_summary, "kmeans_cluster_summary.csv")
write.csv(hc_summary, "hierarchical_cluster_summary.csv")
write.csv(dbscan_summary, "dbscan_cluster_summary.csv")
write.csv(mclust_summary, "mclust_cluster_summary.csv")
write.csv(eval_metrics, "clustering_evaluation_metrics.csv")



#create different clusters for K means 


# Function to perform K-means clustering and visualize results
perform_kmeans_and_summarize <- function(scaled_features, valid_counties, k) {
  # Perform K-means clustering
  kmeans_result <- kmeans(scaled_features, centers = k, nstart = 25)
  
  # Visualize the clusters
  cluster_plot <- fviz_cluster(
    list(data = scaled_features, cluster = kmeans_result$cluster),
    main = paste("K-means Clustering with", k, "Clusters")
  )
  
  # Create cluster summary
  cluster_summary <- create_cluster_summary(kmeans_result$cluster, valid_counties)
  
  # Print cluster summary
  print(paste("K-means Clustering Summary with", k, "Clusters:"))
  print(cluster_summary)
  
  # Return plot and summary
  list(plot = cluster_plot, summary = cluster_summary)
}

# Test for 4, 5, and 6 clusters
results_4_clusters <- perform_kmeans_and_summarize(scaled_features, valid_counties, 4)
results_5_clusters <- perform_kmeans_and_summarize(scaled_features, valid_counties, 5)
results_6_clusters <- perform_kmeans_and_summarize(scaled_features, valid_counties, 6)

# Arrange visualizations for comparison
grid.arrange(results_4_clusters$plot, 
             results_5_clusters$plot, 
             results_6_clusters$plot, 
             ncol = 1)

# Save summaries to CSV
write.csv(results_4_clusters$summary, "kmeans_4_cluster_summary.csv")
write.csv(results_5_clusters$summary, "kmeans_5_cluster_summary.csv")
write.csv(results_6_clusters$summary, "kmeans_6_cluster_summary.csv")

results_4_clusters$plot + ggtitle("K-means Clustering with 4 Clusters")
results_5_clusters$plot + ggtitle("K-means Clustering with 5 Clusters")
results_6_clusters$plot + ggtitle("K-means Clustering with 6 Clusters")



fviz_dend(hc_ward, k = 5, main = "Hierarchical Clustering")
fviz_cluster(list(data = scaled_features, cluster = dbscan_result$cluster), main = "DBSCAN Clustering")
fviz_cluster(list(data = scaled_features, cluster = mclust_clusters), main = "Model-Based Clustering")



# create Silhouette  plots
p_kmeans <- fviz_silhouette(silhouette(kmeans_result$cluster, dist_matrix)) + 
  ggtitle("K-Means Silhouette")

p_hc <- fviz_silhouette(silhouette(hc_clusters, dist_matrix)) + 
  ggtitle("Hierarchical Clustering Silhouette")

p_dbscan <- fviz_silhouette(silhouette(dbscan_result$cluster, dist_matrix)) + 
  ggtitle("DBSCAN Silhouette")

p_mclust <- fviz_silhouette(silhouette(mclust_clusters, dist_matrix)) + 
  ggtitle("Model-Based Clustering Silhouette")

# Arrange all plots in one graph
grid.arrange(p_kmeans, p_hc, p_dbscan, p_mclust, ncol = 2)


#########ground truth

# Categorize deaths_per_1000 into Low, Moderate, High
valid_counties$death_rate_category <- cut(valid_counties$deaths_per_1000,
                                          breaks = quantile(valid_counties$deaths_per_1000, probs = c(0, 0.33, 0.67, 1)),
                                          labels = c("Low", "Moderate", "High"),
                                          include.lowest = TRUE)

# Check distribution of categories
print("Distribution of death rate categories:")
print(table(valid_counties$death_rate_category))

# Function to compare clustering results to ground truth
compare_clusters_to_ground_truth <- function(cluster_assignments, ground_truth) {
  # Ensure both inputs are factors with matching levels
  cluster_assignments <- factor(cluster_assignments)
  ground_truth <- factor(ground_truth)
  
  # Use caret::confusionMatrix to compare predictions and true labels directly
  cm <- caret::confusionMatrix(cluster_assignments, ground_truth)
  
  # Return confusion matrix and F1 scores
  return(list(confusion_matrix = cm$table, f1_scores = cm$byClass[, "F1"]))
}

# K-Means Evaluation
tryCatch({
  kmeans_eval <- compare_clusters_to_ground_truth(kmeans_result$cluster, valid_counties$death_rate_category)
  print("K-Means Evaluation:")
  print(kmeans_eval$confusion_matrix)
  print("F1 Scores:")
  print(kmeans_eval$f1_scores)
}, error = function(e) {
  print("Error in K-Means Evaluation:")
  print(e)
})

# Hierarchical Clustering Evaluation
tryCatch({
  hc_eval <- compare_clusters_to_ground_truth(hc_clusters, valid_counties$death_rate_category)
  print("Hierarchical Clustering Evaluation:")
  print(hc_eval$confusion_matrix)
  print("F1 Scores:")
  print(hc_eval$f1_scores)
}, error = function(e) {
  print("Error in Hierarchical Clustering Evaluation:")
  print(e)
})

# DBSCAN Evaluation (exclude noise cluster 0)
tryCatch({
  dbscan_filtered <- dbscan_result$cluster[dbscan_result$cluster != 0]
  death_rate_filtered <- valid_counties$death_rate_category[dbscan_result$cluster != 0]
  
  dbscan_eval <- compare_clusters_to_ground_truth(dbscan_filtered, death_rate_filtered)
  print("DBSCAN Evaluation (Excluding Noise):")
  print(dbscan_eval$confusion_matrix)
  print("F1 Scores:")
  print(dbscan_eval$f1_scores)
}, error = function(e) {
  print("Error in DBSCAN Evaluation:")
  print(e)
})

# Model-Based Clustering Evaluation
tryCatch({
  mclust_eval <- compare_clusters_to_ground_truth(mclust_clusters, valid_counties$death_rate_category)
  print("Model-Based Clustering Evaluation:")
  print(mclust_eval$confusion_matrix)
  print("F1 Scores:")
  print(mclust_eval$f1_scores)
}, error = function(e) {
  print("Error in Model-Based Clustering Evaluation:")
  print(e)
})
