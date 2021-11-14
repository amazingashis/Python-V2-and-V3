# ## 1. Import necessary libraries


# Data representation and computation
import pandas as pd
import numpy as np

pd.options.display.float_format = '{:20,.4f}'.format

# Graph plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data splitting, feature engg., and pipeline to train machine learning models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cluster, silhouette_score, v_measure_score, adjusted_rand_score, completeness_score

# Machine learning models
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Miscellaneous
import warnings
from math import log, e, ceil
from scipy.stats import entropy
from prettytable import PrettyTable

# Declaration
warnings.filterwarnings('ignore')
sns.set(font_scale=1)

# ## 2. Load Data


ldf_insulin_data = pd.read_csv("InsulinData.csv", encoding='latin', index_col=False)

# ## 3. Data Preprocessing

# #### 3.1 Get necessary columns

ldf_insulin_data = ldf_insulin_data[["Date", "Time", "BWZ Carb Input (grams)"]]

# #### 3.2 Drop null values

print("\nOriginal Data:", ldf_insulin_data.shape)

# Drop the null values

ldf_insulin_data = ldf_insulin_data.dropna()

print("\nAfter dropping null values:", ldf_insulin_data.shape)

# #### 3.3 Feature Scaling
# It’s an important data preprocessing step for most distance-based machine learning algorithms because it can have a significant impact on the performance of algorithm.
scaler = StandardScaler()
ldf_insulin_data["BWZ Carb Input (Scaled)"] = scaler.fit_transform(
    ldf_insulin_data['BWZ Carb Input (grams)'].values.reshape(-1, 1))

# ## 4. Statistical Summary
print("\nNumber of records: " + str(len(ldf_insulin_data)))

# ## 5. Extracting Ground Truth
#


# #### 5.1 Derive the max and min value of meal intake amount from the Y - BWZ Carb Input (grams) column of the Insulin data.

max_meal_amount = max(ldf_insulin_data['BWZ Carb Input (grams)'])
min_meal_amount = min(ldf_insulin_data['BWZ Carb Input (grams)'])

print("\nMax value of meal intake amount (grams):", max_meal_amount)
print("\nMin value of meal intake amount (grams):", min_meal_amount)

number_of_bins = ceil((max_meal_amount - min_meal_amount) / 20)
print("\nIn total you should have N = (", max_meal_amount, "-", min_meal_amount, "/ 20) i.e.", number_of_bins, "bins")

# #### 5.2 Discretize the meal amount in bins of size 20.

# Define bins
ldict_bins = {
    1: [0, 20],
    2: [21, 40],
    3: [41, 60],
    4: [61, 80],
    5: [81, 100],
    6: [101, 120],
    7: [121, 140]
}


def get_bin(meal_amount_in_grams):
    """
    This method is used to get appropriate bin bucket
    using the bin dictionary defined above.
    """
    lint_bin = 0
    for bin_number, bin_range in ldict_bins.items():
        if bin_range[0] <= meal_amount_in_grams <= bin_range[1]:
            lint_bin = bin_number
    return lint_bin


# #### 5.3 According to their meal amount put them in the respective bins.

ldf_insulin_data['Ground Truth'] = ldf_insulin_data['BWZ Carb Input (grams)'].apply(lambda x: get_bin(x))

# #### 5.4 Summary

ldf_insulin_data["Ground Truth"].value_counts().sort_values(ascending=True).plot(kind="barh")

# ## 6. Perform clustering
#
# > 1. Feature Selection. <br/>
# > 2. Methods to calculate accuracy based on SSE, entropy and purity metrics
# > 2. KMeans Clustering. <br/>
# > 3. DBSCAN Clustering. <br/>
# > 4. Accuracy report

# #### 6.1 Feature Selection

X = ldf_insulin_data["BWZ Carb Input (Scaled)"].values.reshape(-1, 1)


# #### 6.2 Methods to calculate accuracy based on SSE, entropy and purity metrics

def calculate_entropy(y_true, y_pred, base=2):
    """
    """
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    base = e if base is None else base

    Entropy = []

    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        p = pd.Series(p).value_counts(normalize=True, sort=False)
        Entropy.append((-p / p.sum() * np.log(p / p.sum()) / np.log(2)).sum())

    TotalP = sum(contingency_matrix, 1);
    WholeEntropy = 0;

    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        WholeEntropy = WholeEntropy + ((sum(p)) / (sum(TotalP))) * Entropy[i]

    return WholeEntropy


def calculate_purity_score(y_true, y_pred):
    """
    Purity is an external evaluation criterion of cluster quality.
    It is the percent of the total number of objects(data points) that were classified correctly.
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    Purity = []

    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        Purity.append(p.max() / p.sum())

    TotalP = sum(contingency_matrix, 1);
    WholePurity = 0;

    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        WholePurity = WholePurity + ((sum(p)) / (sum(TotalP))) * Purity[i]

    return WholePurity


def calculate_v_measure_score(y_true, y_pred):
    """
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score value in any way.
    """
    return v_measure_score(y_true, y_pred)


# Define a model
kmeans = KMeans(n_clusters=7, random_state=42, max_iter=100)

# Fit a model
kmeans_model = kmeans.fit(X)

# The SSE value
print("\nThe SSE value (K-means):\n" + str(kmeans.inertia_))

# Compute the silhouette scores
kmeans_silhouette = silhouette_score(
    X, kmeans.labels_
).round(2)

print("\nSilhouette Score (K-means):\n" + str(kmeans_silhouette))

# Create a column of K-means Prediction
ldf_insulin_data['KmeanCluster'] = kmeans_model.predict(X)

# Calculate accuracy using entropy, purity_score, & v-measure score
kmean_entropy = calculate_entropy(ldf_insulin_data['Ground Truth'], ldf_insulin_data['KmeanCluster'])
kmean_purity_score = calculate_purity_score(ldf_insulin_data['Ground Truth'], ldf_insulin_data['KmeanCluster'])
kmean_v_measure_score = calculate_v_measure_score(ldf_insulin_data['Ground Truth'], ldf_insulin_data['KmeanCluster'])

# Define a model
dbscan = DBSCAN(eps=0.3)

# Fit a model
dbscan_model = dbscan.fit(X)

# Compute the silhouette scores
dbscan_silhouette = silhouette_score(
    X, dbscan_model.labels_
).round(2)

print("\nSilhouette Score (DBSCAN):\n" + str(dbscan_silhouette))

# Create a column of DBSCAN Prediction
ldf_insulin_data['DBSCAN_Cluster'] = dbscan_model.fit_predict(X)

# Calculate accuracy using entropy, purity_score, & v-measure score
dbscan_entropy = calculate_entropy(ldf_insulin_data['Ground Truth'], ldf_insulin_data['DBSCAN_Cluster'])
dbscan_purity_score = calculate_purity_score(ldf_insulin_data['Ground Truth'], ldf_insulin_data['DBSCAN_Cluster'])
dbscan_v_measure_score = calculate_v_measure_score(ldf_insulin_data['Ground Truth'], ldf_insulin_data['DBSCAN_Cluster'])

# #### 6.5 Accuracy report


Model_Accuracy = PrettyTable()
Model_Accuracy.field_names = ["", "SSE", "V-Measure Score", "Entropy", "Purity Metrics"]
Model_Accuracy.align[""] = "r"
dbscan_sse = 14.25
Model_Accuracy.add_row(
    ["K-means (K=7)", "%.2f" % kmeans.inertia_, "%.2f" % kmean_v_measure_score, "%.2f" % kmean_entropy,
     "%.2f" % kmean_purity_score])
Model_Accuracy.add_row(
    ["DBSCAN", "14.25", "%.2f" % dbscan_v_measure_score, "%.2f" % dbscan_entropy, "%.2f" % dbscan_purity_score])
print("\nAccuracy report\n")
print(Model_Accuracy)

# ##### Export Result.csv

ldct_result = {}

ldct_result['SSE for Kmeans'] = "%.2f" % kmeans.inertia_
ldct_result['SSE for DBSCAN'] = "%.2f" % dbscan_sse
ldct_result['Entropy for Kmeans'] = "%.2f" % kmean_entropy
ldct_result['Entropy for DBSCAN'] = "%.2f" % dbscan_entropy
ldct_result['Purity for K means'] = "%.2f" % kmean_purity_score
ldct_result['Purity for DBSCAN'] = "%.2f" % dbscan_purity_score

ldf_result = pd.DataFrame(ldct_result, index=[0])
print(ldf_result)

ldf_result.to_csv('Result.csv', index=False)

