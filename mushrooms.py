import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
import numpy as np


def matrix_multiplication(A, B):
    result = np.empty(shape=A.shape)
    for i in range(len(A)):

        # iterating by column by B
        for j in range(len(B[0])):

            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


data = pd.read_csv('agaricus-lepiota.data')
Class = data.loc[:, "Class"]
data.drop(data.columns[[0]], axis=1, inplace=True)
value = ['?']
data = data[data.isin(value) == False]
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder.fit(data)
transformed_data = encoder.transform(data)
vector_distance = pdist(transformed_data, 'hamming')
distance = squareform(vector_distance)
gaussian_distance = distance  # np.empty(distance.shape)
sigma = 0.1
threshold_distance = math.exp(-np.mean(vector_distance) ** 2 / (2 * sigma ** 2))
D_sqrt = np.empty(shape=gaussian_distance.shape)
L_sn = np.empty(shape=gaussian_distance.shape)

for i in range(gaussian_distance.shape[0]):
    sum_row = 0
    for j in range(gaussian_distance.shape[1]):
        weight = math.exp(-distance[i, j] ** 2 / (2 * sigma ** 2))
        if j != i and weight > threshold_distance:
            gaussian_distance[i, j] = weight
        sum_row += gaussian_distance[i, j]
    D_sqrt[i, i] = 1 / math.sqrt(sum_row)
# print(D_sqrt * gaussian_distance * D_sqrt)
print(D_sqrt.shape)
print(gaussian_distance.shape)
A = matrix_multiplication(D_sqrt, gaussian_distance)
# A = np.dot(D_sqrt, gaussian_distance)
# B = np.matmul(A, D_sqrt)
# L_sn = np.identity(gaussian_distance.shape[0]) - B
# print(L_sn)
# w, v = np.linalg.eig(L_sn)
# print(w)

