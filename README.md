# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.Check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ARUNMOZHI VARMAN T
RegisterNumber:  212223230022
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())

print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:])

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()
```


## Output:
# DATA:
<img width="688" height="208" alt="image" src="https://github.com/user-attachments/assets/5acb4a27-8261-43f0-bc47-1745fbbd368d" />

# ELBOW GRAPH:
<img width="831" height="608" alt="image" src="https://github.com/user-attachments/assets/764f7120-fee8-4c6a-af2c-53c528a5b66f" />

# PREDICTED VALUES:
<img width="725" height="230" alt="image" src="https://github.com/user-attachments/assets/74a66d3c-dad8-40c3-a75e-2037710a7c71" />

# FINAL GRAPH:
<img width="735" height="529" alt="image" src="https://github.com/user-attachments/assets/9d0f222b-00b7-4ca2-9236-2f733121fabe" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
