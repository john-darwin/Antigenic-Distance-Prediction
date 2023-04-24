import numpy as np
import scipy.io as sio
from scipy import sparse
from AANE import AANE
import time
import pandas as pd

'''################# Load data  #################'''
lambd = 1  # the regularization parameter
rho = 5  # the penalty parameter

# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter

'''################# Experimental Settings #################'''
d = 150  # the dimension of the embedding representation
G=pd.read_csv('D:\Code\pycharm\AANE对比实验/network.csv',header=None)
G = sparse.csc_matrix(G)   # Here's the initialization of the sparse matrix.
A=pd.read_excel('D:\Code\pycharm\AANE对比实验\preprocess/attri.xlsx',header=None)
A = sparse.csc_matrix(A)
n = G.shape[0]
Indices = np.random.randint(25, size=n)+1  # 5-fold cross-validation indices

Group1 = []
Group2 = []
[Group1.append(x) for x in range(0, n) if Indices[x] <= 20]  # 2 for 10%, 5 for 25%, 20 for 100% of training group
[Group2.append(x) for x in range(0, n) if Indices[x] >= 21]  # test group
n1 = len(Group1)  # num of nodes in training group
n2 = len(Group2)  # num of nodes in test group
CombG = G[Group1+Group2, :][:, Group1+Group2]
CombA = A[Group1+Group2, :]

'''################# Accelerated Attributed Network Embedding #################'''
print("Accelerated Attributed Network Embedding (AANE), 5-fold with 100% of training is used:")
start_time = time.time()
H_AANE = AANE(CombG, CombA, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))

'''################# AANE for a Pure Network #################'''
print("AANE for a pure network:")
start_time = time.time()
H_Net = AANE(CombG, CombG, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
sio.savemat('Embedding.mat', {"H_AANE": H_AANE, "H_Net": H_Net})
