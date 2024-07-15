#
# Last Modified: Jul 15 2024 <dhananjay.bhaskar@yale.edu>
# Bugfixes by W.A.M. <mohler.william@gmail.com>
#
# CHANGELOG (Version 3.0):
# Corrected integer casts to ensure proper data type conversions
# Fixed issue with incorrect separators in file paths and CSV operations
# No GUI, command-line args for output naming
# Improved error handling and validation for input arguments
#

#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import phate
import scprep

import numpy as np 
import pandas as pd

import scipy as sp
from scipy import special 
from scipy import optimize
from scipy import spatial
from scipy.linalg import svd

from sklearn.decomposition import PCA

# Define helper functions

def scale_down(ideal):
    ideal_list = ideal.copy()
    unique_list = np.unique(ideal)
    list_match = range(len(unique_list))
    
    for r in range(len(ideal_list)):
        ideal_list[r] = list_match[np.where(ideal_list[r] == unique_list)[0][0]]
        
    return np.array(ideal_list)

# Load affinities and convert to kernels

def load_affinities(k_dir_input):
    NxT_input = scprep.io.load_csv(os.path.join(k_dir_input, "cluster_assigments.csv"), cell_names=False).T
    k_list_input = []
    epsilon_list_input = []

    for k in range(1, len(NxT_input.iloc[0, :]) + 1):
        mat = scipy.io.loadmat(os.path.join(k_dir_input, f"{k}-affinity-matrix.mat"))
        A = mat['affinity'].toarray()
        output_dir = os.path.join(k_dir_input, "C-PHATEout")
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(A).to_csv(os.path.join(output_dir, f"{k}-affinities_{inputName}.csv"))
        Q = 1. / np.sum(A, axis=0)
        pd.DataFrame(Q).to_csv(os.path.join(output_dir, f"{k}-Qs_{inputName}.csv"))
        K = np.diag(Q) @ A @ np.diag(Q)  # Creating kernel
        pd.DataFrame(K).to_csv(os.path.join(output_dir, f"{k}-kernels_{inputName}.csv"))

        k_list_input.append(K)
        epsilon_list_input.append(mat['epsilon'][0][0])

    k_list_input.append(np.array([500]))
    return NxT_input, k_list_input, epsilon_list_input

# Main function to run the script

def main():

    arg1 = int(sys.argv[1])
    arg2 = int(sys.argv[2])
    k_dir_input = sys.argv[3]
    argT = int(sys.argv[4])
    argRand = int(sys.argv[5])

    inputName = k_dir_input.split(os.sep)[-2]

    NxT_input, k_list_input, epsilon_list_input = load_affinities(k_dir_input)
    
    sizes_input = [i.shape[0] for i in k_list_input if isinstance(i, np.ndarray)]

    c_phate_input = np.zeros(sum(sizes_input)**2)
    c_phate_input = c_phate_input.reshape(sum(sizes_input),sum(sizes_input))

    # adding kernels to matrix
    base_input = 0

    for i in range(len(sizes_input)):
        c_phate_input[base_input:base_input+sizes_input[i],base_input:base_input+sizes_input[i]] = k_list_input[i]
        base_input = base_input + sizes_input[i]

    ## 1 layer connectivity

    base_input = sizes_input[0]

    for t in range(1,len(sizes_input)-1):
        matching = np.zeros(sizes_input[t+1]*sizes_input[t])
        matching = matching.reshape(sizes_input[t+1],sizes_input[t])
        
        prev = np.array(NxT_input.iloc[:,t-1])   #column t
        fut = np.array(NxT_input.iloc[:,t])
        
        for i in range(len(fut)-1):
            matching[int(fut[i]-1),int(prev[i]-1)] = arg1
            
        c_phate_input[base_input+sizes_input[t] : base_input+sizes_input[t] + sizes_input[t+1], base_input : base_input+sizes_input[t]] = matching
        c_phate_input[ base_input : base_input+sizes_input[t],base_input+sizes_input[t] : base_input+sizes_input[t]+sizes_input[t+1]] = matching.T
            
        base_input = base_input + sizes_input[t]

    ## 2 layer connectivity

    base_input = sizes_input[0]

    for t in range(1,len(sizes_input)-2):
        matching = np.zeros(sizes_input[t+2]*sizes_input[t])
        matching = matching.reshape(sizes_input[t+2],sizes_input[t])
        
        prev = np.array(NxT_input.iloc[:,t-1])   #row t
        fut = np.array(NxT_input.iloc[:,t+1])
        
        for i in range(len(fut)-1):
            matching[int(fut[i]-1),int(prev[i]-1)] = arg2

        c_phate_input[base_input + sizes_input[t]+sizes_input[t+1]:base_input + sizes_input[t]+sizes_input[t+1]+sizes_input[t+2],base_input:base_input+sizes_input[t]] = matching
        c_phate_input[base_input :base_input+sizes_input[t],base_input + sizes_input[t]+sizes_input[t+1]:base_input + sizes_input[t]+sizes_input[t+1]+sizes_input[t+2]] = matching.T
            
        base_input = base_input + sizes_input[t]


    c_phateB_input = np.zeros((sum(sizes_input)-sizes_input[0])**2)
    c_phateB_input = c_phateB_input.reshape(sum(sizes_input)-sizes_input[0],sum(sizes_input)-sizes_input[0])
    c_phateB_input = c_phate_input[sizes_input[0]:sum(sizes_input), sizes_input[0]:sum(sizes_input)]

    pd.DataFrame(c_phateB_input).to_csv(sys.argv[3] + "C-PHATEout/"+"c_phate_"+inputName+".csv")

    ### Visualizing ###
    phate_op = phate.PHATE(n_components=3, t=argT, mds_solver='smacof', n_jobs=-1, random_state=argRand, knn_dist='precomputed_affinity')
    data_phate_input = phate_op.fit_transform(c_phateB_input)

    pd.DataFrame(data_phate_input).to_csv(sys.argv[3] + "C-PHATEout/"+"phateCoords_"+inputName+".csv")

if __name__ == "__main__":
    main()