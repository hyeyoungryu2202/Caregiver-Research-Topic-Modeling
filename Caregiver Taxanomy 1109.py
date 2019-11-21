#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Data for Clustering

# In[1]:


import pandas as pd
topic_distribution_df = pd.read_csv('/Users/angieryu2202/Downloads/ldaoutput_kwphrase_topicdistributions_30_100_2_5000.txt', delimiter = '\t', header = None)


# In[2]:


topic_distribution_df = topic_distribution_df.drop(topic_distribution_df.columns[[0,31]], axis=1)


# In[3]:


topic_distribution_df.head(10)


# In[4]:


topic_distribution_df.tail(10)


# In[5]:


topic_distribution_df.info()


# In[6]:


input_df = pd.read_csv('/Users/angieryu2202/Downloads/ldainput_kwphrase.txt', sep = '\||\t', header = None, engine = 'python')


# In[7]:


input_df = input_df.drop(input_df.columns[0], axis=1)


# In[8]:


input_df.columns = ['journal', 'title', 'year', 'keywords']


# In[9]:


input_df.head(10)


# In[10]:


input_df.tail(10)


# In[11]:


input_df.info()


# In[12]:


caregiver_df = pd.concat([topic_distribution_df, input_df], axis=1)


# In[13]:


caregiver_df.head(10)


# In[14]:


caregiver_df.tail(10)


# In[15]:


caregiver_df.info()


# # K-Means Clustering

# ## (1) Optimal K Calculation

# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.cluster import KMeans


# In[17]:


sum_of_squared_distances = []
K = range(1,80)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(topic_distribution_df)
    sum_of_squared_distances.append(km.inertia_)


# In[18]:


differences = []
i= 0
while i < len(sum_of_squared_distances):
    difference = sum_of_squared_distances[i+1] - sum_of_squared_distances[i]
    print(str(i+1)+"-"+str(i)+":"+str(difference))
    i += 1
    differences.append(difference)
    if i == 78:
        break


# In[19]:


plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# ## (2) KMeans Clustering and Visualization 

# In[20]:


kmeans = KMeans(n_clusters=31)
clusters = kmeans.fit(topic_distribution_df)


# In[21]:


print(kmeans.labels_)


# In[22]:


caregiver_df['kmeans_labels'] = kmeans.labels_


# In[23]:


caregiver_df.head(5)


# In[24]:


caregiver_df['kmeans_labels'].value_counts()


# In[25]:


caregiver_df['kmeans_labels_freq'] = caregiver_df.groupby('kmeans_labels')['kmeans_labels'].transform('count')


# In[26]:


caregiver_df.head(5)


# In[27]:


#centers = np.array(kmeans.cluster_centers_)
#plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
#plt.xlim(1990, 2021)
#plt.scatter(caregiver_df.year, caregiver_df.kmeans_labels, c=caregiver_df.kmeans_labels_freq, cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', label = 'Centroids')
#plt.xlabel('Year', fontsize=12)
#plt.ylabel('Cluster Label', fontsize=12)
#plt.title('K-Means Clusters of Caregiver Research in CS')
#plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=False, ncol=1)


# In[28]:


#plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', label = 'Centroids')
#plt.title('K-Means Clusters of Caregiver Research in CS')
#plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=False, ncol=1)


# In[29]:


import seaborn as sns
plt.title('K-Means Clusters of Caregiver Research in CS')
sns.scatterplot(x=caregiver_df.year,
                y=caregiver_df.kmeans_labels,
                hue=caregiver_df.kmeans_labels_freq)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Cluster Label', fontsize=12)
plt.legend(loc='upper center', bbox_to_anchor=(1.3, 0.9), shadow=False, ncol=1)
#plt.show()
plt.savefig('/Users/angieryu2202/Desktop/kmeans_clusters_visualization_seaborn.png')


# ## (3) PCA

# In[30]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
two_principal_components = pca.fit_transform(topic_distribution_df)
two_principal_df = pd.DataFrame(data = two_principal_components, columns = ['principal component 1', 'principal component 2'])


# In[31]:


final_two_principal_df = pd.concat([two_principal_df, caregiver_df[['kmeans_labels']]], axis = 1)
final_two_principal_df.head(5)


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(two_principal_components[:, 0], two_principal_components[:, 1],
            c=caregiver_df.kmeans_labels, edgecolor='grey', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow', 10))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means 2 Component Clustering PCA')
plt.colorbar();
plt.savefig('/Users/angieryu2202/Desktop/2d_PCA_kmeans.png')


# In[33]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
three_principal_components = pca.fit_transform(topic_distribution_df)
three_principal_df = pd.DataFrame(data = three_principal_components, columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[34]:


final_three_principal_df = pd.concat([three_principal_df, caregiver_df[['kmeans_labels']]], axis = 1)


# In[35]:


final_three_principal_df.head(5)


# In[36]:


labels = []
for label in final_three_principal_df['kmeans_labels']:
    if label not in labels:
        labels.append(label)
from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure(figsize=(17,17))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.set_xlabel('Principal Component 1', fontsize = 30)
ax2.set_ylabel('Principal Component 2', fontsize = 30)
ax2.set_zlabel('Principal Component 3', fontsize = 30)
ax2.set_title('K-Means Clustering 3 Component PCA', fontsize = 30)

colors = ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0",
          "#f0027f","r","#666666", "g", "b",
          "c", "k", "brown", "y", "m",
          "rosybrown", "lightcoral", "purple", "sienna", "darkorange",
          "olivedrab", "darkblue", "blueviolet", "indigo", "slateblue",
          "lime", "limegreen", "peachpuff", "burlywood", "lightseagreen",
         "teal"]
for label, color in zip(labels, colors):
    indicesToKeep = final_three_principal_df['kmeans_labels'] == label
    ax2.scatter(final_three_principal_df.loc[indicesToKeep, 'principal component 1']
               , final_three_principal_df.loc[indicesToKeep, 'principal component 2']
               , final_three_principal_df.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 35)

ax2.legend(labels)
ax2.grid()
fig2.savefig("/Users/angieryu2202/Desktop/3d_PCA_kmeans.png")


# ## (4) Articles Closest to Centroids for Each Cluster 

# In[37]:


import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
#km = KMeans(n_clusters=31).fit(topic_distribution_df)
# This function computes for each row in X, the index of the row of Y which is closest (according to the specified distance). The minimal distances are also returned
kmeans_closest, _ = pairwise_distances_argmin_min(clusters.cluster_centers_, topic_distribution_df)
print(kmeans_closest)


# In[38]:


kmeans_closest_titles = []
for index in kmeans_closest:
    kmeans_closest_titles.append(caregiver_df.title[index])
    print(str(index)+": "+str(caregiver_df.title[index]))


# In[39]:


kmeans_clusters_range = list(range(31))
kmeans_closest_df = pd.DataFrame(list(zip(kmeans_clusters_range, kmeans_closest, kmeans_closest_titles)), columns =['kmeans_cluster_number', 'kmeans_closest_sample_index', 'kmeans_closest_sample_title']) 


# In[40]:


kmeans_closest_df.head(10)


# In[41]:


kmeans_closest_df.tail(10)


# # Hierarchical Clustering

# ## (1) Optimal Number of Clusters 

# In[42]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Article Dendograms")
dend = shc.dendrogram(shc.linkage(topic_distribution_df, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')


# ## (2) Agglomerative Clustering and Visualization 

# In[43]:


import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[44]:


from sklearn.cluster import AgglomerativeClustering
# number of clusters, affinity (distance between the datapoints), linkage = ward (minimizes the variant between the clusters)
agg_cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
agg_clusters = agg_cluster.fit_predict(topic_distribution_df)
print(agg_cluster.labels_)


# In[45]:


caregiver_df['hierarchical_cluster'] = agg_cluster.labels_


# In[46]:


caregiver_df['hierarchical_cluster'].value_counts()


# In[47]:


caregiver_df['hierarchical_cluster_freq'] = caregiver_df.groupby('hierarchical_cluster')['hierarchical_cluster'].transform('count')
caregiver_df.head(5)


# In[48]:


#plt.scatter(caregiver_df.year, caregiver_df.max_value, c=cluster.labels_, cmap='rainbow')
plt.title('Hierarchical Clusters of Caregiver Research in CS')
plt.scatter(caregiver_df.year, # x
           caregiver_df.hierarchical_cluster, # y
           alpha=0.2,
           c=caregiver_df.hierarchical_cluster_freq, # marker color
            cmap='viridis')
plt.xlim(1990, 2021)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Cluster Label', fontsize=12)
plt.colorbar()
#plt.show()
plt.savefig('/Users/angieryu2202/Desktop/hierarchical_clusters_visualization_matplotlib.png')


# In[49]:


import seaborn as sns
plt.title('Hierarchical Clusters of Caregiver Research in CS')
sns.scatterplot(x=caregiver_df.year,
                y=caregiver_df.hierarchical_cluster,
                hue=caregiver_df.hierarchical_cluster_freq)
plt.xlim(1990, 2021)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Cluster Label', fontsize=12)
plt.legend(loc='upper center', bbox_to_anchor=(1.3, 0.7), shadow=False, ncol=1)
#plt.show()
plt.savefig('/Users/angieryu2202/Desktop/hierarchical_clusters_visualization_seaborn.png')


# ## (3) PCA

# In[50]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
two_principal_components = pca.fit_transform(topic_distribution_df)
two_hierarchical_principal_df = pd.DataFrame(data = two_principal_components, columns = ['principal component 1', 'principal component 2'])


# In[51]:


final_two_hierarchical_principal_df = pd.concat([two_hierarchical_principal_df, caregiver_df[['hierarchical_cluster']]], axis = 1)
final_two_hierarchical_principal_df.head(5)


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(two_principal_components[:, 0], two_principal_components[:, 1],
            c=caregiver_df.hierarchical_cluster, edgecolor='grey', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow', 10))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Hierarchical Clustering 2 Component PCA')
plt.colorbar();
plt.savefig('/Users/angieryu2202/Desktop/2d_PCA_hierarchical.png')


# In[53]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
three_principal_components = pca.fit_transform(topic_distribution_df)
three_hierarchical_principal_df = pd.DataFrame(data = three_principal_components, columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[54]:


final_three_hierarchical_principal_df = pd.concat([three_hierarchical_principal_df, caregiver_df[['hierarchical_cluster']]], axis = 1)
final_three_hierarchical_principal_df.head(5)


# In[55]:


labels = []
for label in final_three_hierarchical_principal_df['hierarchical_cluster']:
    if label not in labels:
        labels.append(label)
from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure(figsize=(17,17))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.set_xlabel('Principal Component 1', fontsize = 30)
ax2.set_ylabel('Principal Component 2', fontsize = 30)
ax2.set_zlabel('Principal Component 3', fontsize = 30)
ax2.set_title('Hierarchical Clustering 3 Component PCA', fontsize = 30)

colors = ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0",
          "#f0027f","r","#666666", "g", "b",
          "c", "k", "brown", "y", "m",
          "rosybrown", "lightcoral", "purple", "sienna", "darkorange",
          "olivedrab", "darkblue", "blueviolet", "indigo", "slateblue",
          "lime", "limegreen", "peachpuff", "burlywood", "lightseagreen",
         "teal"]
for label, color in zip(labels, colors):
    indicesToKeep = final_three_hierarchical_principal_df['hierarchical_cluster'] == label
    ax2.scatter(final_three_hierarchical_principal_df.loc[indicesToKeep, 'principal component 1']
               , final_three_hierarchical_principal_df.loc[indicesToKeep, 'principal component 2']
               , final_three_hierarchical_principal_df.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 35)

ax2.legend(labels)
ax2.grid()
fig2.savefig("/Users/angieryu2202/Desktop/3d_PCA_hierarchical.png")


# ## (4) Articles Closest to Centroids for Each Cluster 

# In[56]:


topic_distribution_df.groupby(caregiver_df["hierarchical_cluster"]).mean()


# In[57]:


import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
# This function computes for each row in X, the index of the row of Y which is closest (according to the specified distance). The minimal distances are also returned
agg_closest, _ = pairwise_distances_argmin_min(topic_distribution_df.groupby(caregiver_df["hierarchical_cluster"]).mean(), topic_distribution_df)
print(agg_closest)


# In[58]:


agg_closest_titles = []
for index in agg_closest:
    agg_closest_titles.append(caregiver_df.title[index])
    print(str(index)+": "+str(caregiver_df.title[index]))


# In[59]:


agg_clusters_range = list(range(7))
agg_closest_df = pd.DataFrame(list(zip(agg_clusters_range, agg_closest, agg_closest_titles)), columns =['agg_cluster_number', 'agg_closest_sample_index', 'agg_closest_sample_title']) 


# In[60]:


agg_closest_df


# # Save the three dataframes (caregiver_df, kmeans_closest_df, agg_closest_df)

# In[61]:


caregiver_df.to_csv('/Users/angieryu2202/Desktop/caregiver_df.csv')
kmeans_closest_df.to_csv('/Users/angieryu2202/Desktop/kmeans_closest_df.csv')
agg_closest_df.to_csv('/Users/angieryu2202/Desktop/agg_closest_df.csv')

