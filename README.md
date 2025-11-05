# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset
2.Plot the graph using elbow method
3.Print the predicted array
4.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: KARTHICK KISHORE T
RegisterNumber: 212223220042
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers (1).csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
<img width="580" height="443" alt="image" src="https://github.com/user-attachments/assets/cb125fec-1f3e-4af8-a8e8-086c38afad54" />
<img width="817" height="443" alt="image" src="https://github.com/user-attachments/assets/19b89916-4004-4b85-b9c7-5f0750370788" />
<img width="737" height="544" alt="image" src="https://github.com/user-attachments/assets/02e62e1d-5050-47f1-ada2-d0177b30b78a" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
