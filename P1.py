import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# importing k-mean
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv")
# print(data.shape)
data.drop(['S.NO'], axis=1, inplace=True)
X = data


le = LabelEncoder()
X['part_time_job'] = le.fit_transform(X['part_time_job'])
X['Study-hours'] = le.fit_transform(X['Study-hours'])
X['social_media_usage'] = le.fit_transform(X['social_media_usage'])
X['Residence'] = le.fit_transform(X['Residence'])
X['chosen_field'] = le.fit_transform(X['chosen_field'])
X['Satisfied_with_educational_system_and_job_market'] = le.fit_transform(X['Satisfied_with_educational_system_and_job_market'])
X['family_responsibilities'] = le.fit_transform(X['family_responsibilities'])


# y = data['smoking']

# y = le.transform(y)
# print(X.head())


# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select the features to use for normalization
# X = data[["Study-hours", "part_time_job","social_media_usage","Residence","chosen_field","Satisfied_with_educational_system_and_job_market","family_responsibilities","CGPA"]]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a k-means model to the standardized data
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled)

# Use the k-means model to transform the data
X_normalized = kmeans.transform(X_scaled) 


############################### Elbow plot #################################
################### To check how many clusters to make #####################

# inertias = []
# # Fit a KMeans model for k = 1 to k = 10 and store the values of kmeans.inertia_
# for k in range(1, 11):
#     model = KMeans(n_clusters=k)
#     model.fit(X)
#     inertias.append(model.inertia_)

# # Plot the values of kmeans.inertia_ for different values of k
# plt.plot(range(1, 11), inertias, '-o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.show()
#############################################################################

kmeans = KMeans(n_clusters=3) 
# fitting the values
kmeans.fit(X)


# print(kmeans.cluster_centers_)
# print (kmeans.inertia_)

cluster_labels = kmeans.labels_
# Create a dictionary to store the data points for each cluster
clusters = {}

# Iterate over the data points and their cluster labels
for i, label in enumerate(cluster_labels):
  # If the cluster label is not in the dictionary, add it as a key
  if label not in clusters:
    clusters[label] = []

  # Append the data point to the appropriate cluster
  clusters[label].append(X.iloc[i])

# Iterate over the clusters and write the data points to separate files
for label, data in clusters.items():
  with open('cluster_{}.csv'.format(label), 'w') as f:
    for point in data:
      f.write(','.join(str(x) for x in point) + '\n')
      print('Cluster {} written to file'.format(label))
      # Assume that you have a data point called "point"
      # that you want to check which cluster it belongs to

      # prediction = kmeans.predict(point)
      # cluster_label = prediction[100]
      # print("The data point belongs to cluster: {}".format(cluster_label))
      # Assume that you have a data point called "point"
      # that you want to check which cluster it belongs to

      # distances = kmeans.transform(point)
      # cluster_label = np.argmin(distances)
      # print("The data point belongs to cluster: {}".format(cluster_label))



