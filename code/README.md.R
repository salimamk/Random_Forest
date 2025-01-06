#Load the ISlR2 Package
install.packages("ISLR2")
library(ISLR2)

#Load the Hitters dataset
?Hitters
data(Hitters)
View(Hitters)

#Q1: Prepare the dataset and set seed =1
#Prepare the data
# Remove rows with missing values
Hitters_clean <- na.omit(Hitters)

# Create logSalary column by replacing Salary
Hitters_clean$logSalary <- log(Hitters_clean$Salary)

# Remove the original Salary column
Hitters_clean$Salary <- NULL

# Check the structure of the cleaned dataset
str(Hitters_clean)
View(Hitters_clean)

# Set the seed for reproducibility
set.seed(1)

#Q2: Partition your data into a 75% training set and a 25% test set (hold-out set).

# Create a 75-25 train-test split
train_indices <- sample(1:nrow(Hitters_clean), size = 0.75 * nrow(Hitters_clean))

# Split the dataset into training and test sets
train_data <- Hitters_clean[train_indices, ]
test_data <- Hitters_clean[-train_indices, ]

# Check the dimensions of the sets
dim(train_data)
#Ans: 197 20
dim(test_data)
#Ans: 66 20

#Q3:Apply bagging to the training set and then predict logSalary for the test set. 
#Report the mean squared error (MSE) for the test set.
# Load the randomForest package
install.packages("randomForest")
library(randomForest)

# Apply bagging (mtry = number of predictors)
bagging_model <- randomForest(logSalary ~ ., data = train_data, mtry = ncol(train_data) - 1)

# Predict logSalary on the test set
bagging_predictions <- predict(bagging_model, newdata = test_data)

# Calculate the Mean Squared Error (MSE)
bagging_mse <- mean((bagging_predictions - test_data$logSalary)^2)
bagging_mse
#Ans: MSE = 0.2353
# #Interpretation:The bagging model, which uses all predictors for splitting, has a mean squared error (MSE) of 0.2353. 
# This relatively low MSE suggests that the model is able to make reasonably accurate predictions for the logSalary, 
# but there may still be room for improvement compared to other models, such as random forests or boosting.

#Q4:Apply random forests to the training set and then predict logSalary for the test set. 
#Use three sensible values of m (number of predictor variables available to split on) 
#and report test error (MSE) for all three m, stating which one is the best.

# Define values for m (number of predictors to use for splitting)
m_values <- c(5, 8, 12)

# Initialize an empty vector to store MSE for each value of m
mse_rf <- numeric(length(m_values))

# Apply random forests for different values of m
for (i in 1:length(m_values)) {
  rf_model <- randomForest(logSalary ~ ., data = train_data, mtry = m_values[i])
  rf_predictions <- predict(rf_model, newdata = test_data)
  mse_rf[i] <- mean((rf_predictions - test_data$logSalary)^2)
}

# Show MSE for each m
mse_rf

#Ans:0.2241242 0.2343225 0.2368743
# Interpretation: The best value for m is 5, which leads to the lowest MSE of 0.2241. 
# This means that using 5 predictor variables for splitting leads to the most accurate predictions among the three tested values. Larger values of m, particularly 8 and 12, result in slightly higher MSE, 
# indicating that smaller m may be more effective in reducing model error.

#Q5:Apply boosting to the training set and then predict logSalary for the test set. 
#Use three different values for tree depth (a.k.a. interaction depth). 
#Report all three results (MSE), stating which result is the best.

# Load the gbm package for boosting
install.packages("gbm")
library(gbm)

# Define values for tree depth (interaction depth)
tree_depths <- c(2, 3, 4)

# Initialize an empty vector to store MSE for each tree depth
mse_boosting <- numeric(length(tree_depths))

# Apply boosting for different values of tree depth
for (i in 1:length(tree_depths)) {
  boosting_model <- gbm(logSalary ~ ., data = train_data, distribution = "gaussian", 
                        n.trees = 1000, interaction.depth = tree_depths[i])
  boosting_predictions <- predict(boosting_model, newdata = test_data, n.trees = 1000)
  mse_boosting[i] <- mean((boosting_predictions - test_data$logSalary)^2)
}

# Show MSE for each tree depth
mse_boosting

#Ans: 0.3479422 0.3676844 0.3496743
# Interpretation:The best tree depth is 2, with an MSE of 0.3479. 
# This indicates that a shallower tree depth (fewer interactions between variables) results in the best model performance for predicting logSalary. Deeper trees (3 and 4) have higher MSE, 
# which might suggest overfitting or less generalization to unseen data.

#Q6:For the best random forest (i.e., for the best value of m), produce a Variable Importance Plot. 
#Two graphs will be produced. With reference to the graph with %IncMSE on the x-axis, 
#write a paragraph (at least two sentences) explaining what the graph means
#and summarizing the results concerning the dataset under study.

# Fit the random forest model with the best 'm' (e.g., m = 8)
best_rf_model <- randomForest(logSalary ~ ., data = train_data, mtry = 8, importance = TRUE)

# Set the file path for saving the plots
file_path <- "figures/var_importance_plots.png"

# Save the two variable importance plots as a PNG file
png(file_path, width = 1200, height = 600)

# Set up the layout for two side-by-side plots
par(mfrow = c(1, 2))

# Plot 1: Mean Decrease Accuracy
varImpPlot(best_rf_model, type = 1, main = "Mean Decrease Accuracy")

# Plot 2: Mean Decrease Gini
varImpPlot(best_rf_model, type = 2, main = "Mean Decrease Gini")

# Close the PNG device to save the plots
dev.off()

# Confirm the plots have been saved
file.exists(file_path)

# View the plots in the R console
varImpPlot(best_rf_model, type = 1)  # Mean Decrease Accuracy
varImpPlot(best_rf_model, type = 2)  # Mean Decrease Gini

# Interpretation:The graph with %IncMSE on the x-axis shows the importance of each predictor variable, 
# where variables further to the right contribute more to reducing the model's error (MSE). In the Hitters dataset, this indicates that the variables on the right are crucial for accurately predicting logSalary, 
# while those on the left have a minimal impact on the model’s performance.










