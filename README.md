# Email-Spam-Detection
Table of Contents
1. Introduction

2. Scope of the Analysis

3. Existing System
i.	Drawbacks or limitations of existing system

4. Source of dataset

5. ETL process

6. Analysis on dataset (for each analysis)
i.	Introduction
ii.	General Description
iii.	Specific Requirements, functions and formulas
iv.	Analysis results
v.	Visualization   (Dashboard) 

  7. List of Analysis with results

8. Future scope

9. Linkedin Post

10. References






Project Report on Predictive Analysis for Spam Detection
1. Introduction
This project focuses on creating a predictive model to classify emails as "Spam" or "Not Spam" using machine learning algorithms, including Naive Bayes, Support Vector Machine (SVM), and Decision Trees. The aim is to enhance email filtering systems, which is essential for improving user productivity and maintaining security against unwanted content.

2. Scope of Analysis
•	Objective: Develop an efficient machine learning-based spam detection system.
•	Applications: Improved spam filtering for enterprise and personal email systems.
•	Technological Improvements: This analysis could extend to real-time detection capabilities and integration with larger or dynamic datasets.

3. Dataset and Preprocessing
Source of Dataset
The dataset consists of labeled emails, either as "Spam" or "Not Spam". Typical datasets include sources such as the Enron Email Dataset or Kaggle's spam dataset(Project_INT234).


ETL Process
•	Extract: Load the dataset using read.csv() and inspect using str() and View().
 
•	Transform:
o	Convert text to lowercase to standardize data.
o	Remove punctuation, numbers, and stop words to reduce noise.
o	Apply stemming for word base normalization.
•	Load: Create a Document-Term Matrix (DTM) and remove sparse terms to maintain manageable feature sets.
•	
•	 
4. Analysis on Dataset
 
 
Data Preparation
The text data was preprocessed to form a DTM. This transformation allowed the text to be converted into a matrix format suitable for machine learning algorithms.

Models Used
•	Naive Bayes: A probabilistic model effective for text classification.

•	SVM: Utilizes a hyperplane to separate spam from non-spam.


•	Decision Tree: A tree-structured model that splits data based on feature values.

Evaluation Metrics
Confusion matrices were used to evaluate the models. Each confusion matrix provided:
•	True Positives (TP): Correctly classified spam emails.
•	True Negatives (TN): Correctly classified non-spam emails.
•	False Positives (FP): Non-spam emails incorrectly classified as spam.
•	False Negatives (FN): Spam emails incorrectly classified as nonspam.
•	 
Results Overview
•	Naive Bayes: Delivered reliable performance, as indicated by its confusion matrix and ROC curve.
 


•	SVM: Generally performed well, showing robustness in distinguishing classes.
 


•	Decision Tree: Effective but may exhibit overfitting if not pruned or tuned adequately.
 
Visualization
•	Top 10 Frequent Words: A bar plot displaying the most common terms in the dataset.
•	Word Cloud: Depicts the top 50 words for a visual summary of frequent terms.
•	ROC Curves: Used to compare model performances visually. SVM and Naive Bayes models are assessed based on the Area Under the Curve (AUC), with higher values indicating superior performance.



5. Future Scope
•	Deep Learning Integration: Advanced models like LSTM (Long Short-Term Memory) networks could improve prediction accuracy by learning sequential text patterns(Project_INT234).
•	Real-Time Implementation: Deploying the spam detection model in live email systems to classify emails as they arrive and updating the model based on user feedback for evolving spam trends.
6. Conclusion
This project demonstrated the potential of machine learning models such as Naive Bayes, SVM, and Decision Trees in building a robust spam detection system. The models, especially SVM and Naive Bayes, were effective, with future opportunities including integrating deep learning and real-time processing for enhanced accuracy and adaptability.

References
•	Data.gov.in
•	chatGPT
•	GOOGLE




