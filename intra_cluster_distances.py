import pandas as pd
import numpy as np

def writeDistanceToCSV(disMatPath):
    """
    This function will take the path to the distance matrix that is exported from ML-DSP and 
    will create a .csv file containing the intra-cluster distances for each cluster. 
    
    :param disMatPath: Path to distance matrix from ML-DSP
    """
    # Read in data
    data = (pd.read_csv(disMatPath, header=None).drop(axis=0, index=1)
                    .drop(1, axis=1))
    # Get cluster labels from data 
    cluster_labels = data.iloc[0, 1:].unique()
    # initialize dataframe for output to .csv
    outData = pd.DataFrame()
    # for every cluster label 
    for label in cluster_labels:
        
        # extract the distance matrix for the cluster
        clusterDisMat = (data.loc[data.eq(label).any(), data.eq(label).any()]
                                  .set_index(0).drop('ClusterName', axis=0).values
                                  .astype(float))
    # get the indices for the upper triangular of our distance matrix
        upTriInd = np.triu_indices(len(clusterDisMat))
    
    # get the values for the upper triangular
        clusterDisMat = clusterDisMat[upTriInd]
    
    # remove all zeroes from data (these are the values on diagonal which we don't need)
        clusterData = clusterDisMat[clusterDisMat != 0]
        # put data into Series
        clusterData = pd.Series(clusterData)
        # add Series to DataFrame for output to .csv
        outData = pd.concat([outData, clusterData], ignore_index=True, axis=1)
    # set column names to cluster labels
    outData.columns = cluster_labels
    
    # write to csv
    outData.to_csv('clusterDistances.csv', index=False)