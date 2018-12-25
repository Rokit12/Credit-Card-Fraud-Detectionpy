#!/usr/bin/env python
# coding: utf-8

# # # Credit Card Fraud Detection
# 
# Presented by Robinson Muiru!
# 
# Throughout the financial sector, machine learning algorithms are being developed to detect fraudulent transactions. In this project, that is exactly what we are going to be doing as well. Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, we are going to identify transactions with a high probability of being credit card fraud. In this project, we will build and deploy the following two machine learning algorithms:
#           Local Outlier Factor (LOF)
#           Isolation Forest Algorithm
# Furthermore, using metrics suchs as precision, recall, and F1-scores, we will investigate why the classification accuracy for these algorithms can be misleading.
# In addition, we will explore the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in our data set. Let's get started!
# 1. Importing Necessary Libraries
# To start, let's print out the version numbers of all the libraries we will be using in this project. This serves two purposes - it ensures we have installed the libraries correctly and ensures that this tutorial will be reproducible. 
# 

# In[8]:


import sys
import pandas
import numpy
import matplotlib
import sklearn
import scipy


print('python: {}'.format(sys.version))
print('pandas: {}'.format(pandas.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Load the Data Set

# In[10]:


import pandas as pd
data = pd.read_csv(r'C:\Users\Robinson\Desktop\creditcard.csv')


# In[38]:


#Start exploring the dataset
print(data.columns)


# In[11]:


#print the shape of the data
data = data.sample(frac= 0.1, random_state = 1)
print(data.shape)
print(data.describe())


# In[12]:


#plot the hitogram of each parameter
data.hist(figsize = (20, 20))
plt.show()


# In[15]:


#Determine the number of Fraud cases in the dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid transactions: {}'.format(len(data[data['Class'] == 0])))


# In[1]:


#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax= .8, square= True)
plt.show()


# In[2]:


#Get all the columns from the Dataframe
columns = data.columns.tolist()

#Filter the columns to remove data we dont want
columns = [c for c in columns if c not in ["Class"]]

#store the variable we will be predicting on 
Target = "Class"

X = data[columns]
Y = data[Target]

#Print the shapes
print(X.shape)
print(Y.shape)


# 3. Unsupervised Outlier Detection
# Now that we have processed our data, we can begin deploying our machine learning algorithms. We will use the following techniques: 
# Local Outlier Factor (LOF)
# The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.
# Isolation Forest Algorithm
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

# In[3]:


from sklearn.metrics import classification_reports, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

#Define random state
state = 1

#define the outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}


# In[ ]:


#fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

