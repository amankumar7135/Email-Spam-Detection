#===============================================================================
#========================PROJECT_PREDICTIVE_ANALYSIS============================
#===============================================================================

# Load all necessary libraries
library(tm)
library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(pROC)
library(wordcloud)
library(rpart)
library(RColorBrewer)

# Load the dataset
data <- read.csv(file.choose(), stringsAsFactors = FALSE)
View(data)
str(data)

# Convert spam column to factor
data$spam <- factor(data$spam, levels = c(0, 1), labels = c("Not_Spam", "Spam"))

# Plot class distribution
ggplot(data, aes(x = spam)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Class Distribution", x = "Spam/Not Spam", y = "Count") +
  theme_minimal()

# Text preprocessing
corpus <- VCorpus(VectorSource(data$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stemDocument)

# Convert to Document-Term Matrix and Remove Sparse Terms
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)
data_cleaned <- as.data.frame(as.matrix(dtm))
data_cleaned$spam <- data$spam  # Add label column back

# Visualize Top 10 Most Frequent Words
word_freq <- sort(colSums(as.matrix(dtm)), decreasing = TRUE)
top_words <- data.frame(word = names(word_freq), freq = word_freq) %>% head(10)
ggplot(top_words, aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Top 10 Frequent Words", x = "Words", y = "Frequency") +
  theme_minimal()

# Word cloud for top 50 frequent words
wordcloud(corpus, min.freq = 50, random.order = FALSE, colors = brewer.pal(5, "Dark2"))

# Split Data into Training and Testing
set.seed(123)
trainIndex <- createDataPartition(data_cleaned$spam, p = 0.8, list = FALSE)
train_data <- data_cleaned[trainIndex, ]
test_data <- data_cleaned[-trainIndex, ]

# Naive Bayes model
model <- naiveBayes(spam ~ ., data = train_data)

# SVM model
svm_model <- svm(spam ~ ., data = train_data, kernel = "linear", cost = 1, scale = FALSE, probability = TRUE)

# Decision Tree model
tree_model <- rpart(spam ~ ., data = train_data, method = "class")

# Make predictions and evaluate Naive Bayes
model_predictions <- predict(model, test_data)
model_confusion_matrix <- confusionMatrix(model_predictions, test_data$spam)
print("Naive Bayes Confusion Matrix:")
print(model_confusion_matrix)

# Plot Naive Bayes Confusion Matrix
nb_cm <- as.data.frame(model_confusion_matrix$table)
colnames(nb_cm) <- c("Actual", "Predicted", "Freq")
ggplot(nb_cm, aes(Actual, Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Naive Bayes Confusion Matrix", x = "Actual", y = "Predicted") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()

# Make predictions and evaluate SVM
svm_predictions <- predict(svm_model, test_data)
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$spam)
print("SVM Confusion Matrix:")
print(svm_conf_matrix)

# Plot SVM Confusion Matrix
svm_cm <- as.data.frame(svm_conf_matrix$table)
colnames(svm_cm) <- c("Actual", "Predicted", "Freq")
ggplot(svm_cm, aes(Actual, Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "SVM Confusion Matrix", x = "Actual", y = "Predicted") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()

# Make predictions and evaluate Decision Tree
tree_predictions <- predict(tree_model, test_data, type = "class")
tree_conf_matrix <- confusionMatrix(tree_predictions, test_data$spam)
print("Decision Tree Confusion Matrix:")
print(tree_conf_matrix)

# Plot Decision Tree Confusion Matrix
tree_cm <- as.data.frame(tree_conf_matrix$table)
colnames(tree_cm) <- c("Actual", "Predicted", "Freq")
ggplot(tree_cm, aes(Actual, Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Decision Tree Confusion Matrix", x = "Actual", y = "Predicted") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()

# ROC Curve for Naive Bayes
model_probability <- predict(model, test_data, type = "raw")[,2]
print(head(model_probability))  # Debug print
roc_model <- roc(as.numeric(test_data$spam) - 1, model_probability)

# ROC Curve for SVM
svm_prob <- attr(predict(svm_model, test_data, probability = TRUE), "probabilities")[,2]
print(head(svm_prob))  # Debug print
roc_svm <- roc(as.numeric(test_data$spam) - 1, svm_prob)

# ROC Curve for Decision Tree
tree_prob <- predict(tree_model, test_data, type = "prob")[,2]
print(head(tree_prob))  # Debug print
roc_tree <- roc(as.numeric(test_data$spam) - 1, tree_prob)

# Plot ROC curves for Naive Bayes, SVM, and Decision Tree
plot(roc_model, col = "blue", main = "ROC Curve for Naive Bayes, SVM, and Decision Tree", lwd = 2)
lines(roc_svm, col = "red", lwd = 2)
lines(roc_tree, col = "green", lwd = 2)
legend("bottomright", legend = c("Naive Bayes", "SVM", "Decision Tree"), col = c("blue", "red", "green"), lwd = 2)
