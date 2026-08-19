import pandas as pd
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from sknetwork.classification import DiffusionClassifier
from sknetwork.regression import Dirichlet
from projectframe import ProjectFrame
from joblib import Parallel, delayed
from intervalframe import IntervalFrame
import os

# Local imports
from ...data.import_data import get_data_file
from ..regions.methyl_regions import methyl_blocks


class BLOCKimputer(object):
    """
    """
    
    def __init__(self,
                 network_file="MPACT_blocks_hg38_network.h5",
                 features_file = "MPACT_block_features.parquet"):
        """
        """

        # Get File
        if not os.path.exists(network_file):
            network_file = get_data_file(network_file)
        if not os.path.exists(features_file):
            features_file = get_data_file(features_file)
        
        # Read imputation matrix
        f = h5py.File(network_file, "r")
        self.features = IntervalFrame.read_parquet(features_file)
        data = np.array(f["data"])
        indices = np.array(f["indices"])
        indptr = np.array(f["indptr"])
        shape = np.array(f["shape"])
        shape = (shape[0], shape[1])
        f.close()
        self.network = csr_matrix((data, indices, indptr), shape=shape)
        
        
    def match_matrix(self, X: IntervalFrame):
        """
        """

        # Check if the features match
        if X.shape[0] == self.features.shape[0]:
            return X
        
        # Match features
        blocks = methyl_blocks(X, self.features)
        
        return blocks
        
    
    def impute_status(self, X, threshold=0.75):
        """
        """

        # Initialize the prediction matrix
        prediction = X
    
        # Fit the Diffusion model
        diffusion = DiffusionClassifier()
    
        # Iterate over the columns
        for n in range(prediction.shape[0]):            
            #print(n, flush=True)

            # Predict the missing values
            x = prediction.values[n,:]
            is_null = pd.isnull(x)
            if np.sum(~is_null) > 10:
                labels = {i:x[i] for i in range(len(x)) if np.isnan(x[i]) == False}
                labels_pred = diffusion.fit_predict(self.network, labels)
                probs = diffusion.predict_proba().max(axis=1)

                confident = np.logical_and(is_null, probs >= threshold)
                prediction.iloc[n,confident] = labels_pred[confident]

        return prediction
        
    
    def impute_beta(self,
                    X: pd.core.frame.DataFrame,
                    verbose: bool = False):
        """
        Predict the missing values in X using the network A.

        Parameters
        ----------
            X : pd.DataFrame
                The data matrix.
            verbose : bool
                If True, print progress.

        Returns
        -------
            prediction : pd.DataFrame
                The predicted values.
        """

        # Fit the Dirichlet model
        dirichlet = Dirichlet()
        prediction = X

        # Iterate over the columns
        output_values = prediction.values
        for n in range(prediction.shape[0]):
            if verbose: print(n, flush=True)

            # Predict the missing values
            x = prediction.values[n,:]
            if np.sum(~pd.isnull(x)) > 10:
                #values = {i:x[i] for i in range(len(x)) if np.isnan(x[i]) == False}
                values = {i: x[i] for i in np.where(~np.isnan(x))[0]}
                values_pred = dirichlet.fit_predict(self.network, values)
            else:
                values_pred = np.repeat(np.nan, len(x))

            output_values[n,:] = values_pred
        
        prediction.loc[:,:] = output_values

        return prediction


def chunkify(data, chunks):
    return np.array_split(data, chunks)

def parallel_impute_status(imputer, data, threshold, n_jobs=-1, chunks=10):
    data_chunks = chunkify(data, chunks)
    results = Parallel(n_jobs=n_jobs)(delayed(imputer.impute_status)(chunk, threshold) for chunk in data_chunks)
    return pd.concat(results, axis=0)

def parallel_impute_beta(imputer, data, n_jobs=-1, verbose=False, chunks=10):
    data_chunks = chunkify(data, chunks)
    results = Parallel(n_jobs=n_jobs)(delayed(imputer.impute_beta)(chunk, verbose) for chunk in data_chunks)
    return pd.concat(results, axis=0)


def impute_blocks(data: pd.DataFrame,
                network_file: str = "MPACT_blocks_hg38_network.h5",
                features_file: str = "MPACT_block_features.parquet",
                method: str = "binary",
                threshold: float = 0.75,
                n_imputes: int = 5,
                fill: bool = False,
                n_jobs: int = -1,
                chunks: int = 10):
    """
    Perform imputation using multiprocessing in chunks to optimize memory usage.
    
    Parameters
    ----------
        pf : ProjectFrame
            ProjectFrame object containing data.
        network_file : str
            Path to the network file.
        method : str
            "binary" for binary classification or "continuous" for Dirichlet regression.
        threshold : float
            Confidence threshold for binary imputation.
        n_imputes : int
            Number of iterative imputations.
        fill : bool
            Whether to fill remaining missing values with threshold=0.0.
        n_jobs : int
            Number of parallel jobs (-1 uses all available cores).
        chunks : int
            Number of chunks to split data for parallel processing.
    
    Returns
    -------
        pf : ProjectFrame
            Updated ProjectFrame with imputed values.
    """
    
    # Read network
    imputer = BLOCKimputer(network_file=network_file,
                           features_file=features_file)
    
    # Match features
    data = imputer.match_matrix(data)
    data = data.df.T
    
    if method == "binary":
        data[data > 0.6] = 1
        data[data <= 0.6] = 0

        # Impute in parallel using chunks
        for _ in range(n_imputes):
            data = parallel_impute_status(imputer, data, threshold, n_jobs, chunks)
        
        if fill:
            data = parallel_impute_status(imputer, data, threshold=0.0, n_jobs=n_jobs, chunks=chunks)
        
        # Format
        data[data == 0] = -1
        data[pd.isnull(data)] = 0
    
    else:
        # Continuous imputation in parallel using chunks
        data = parallel_impute_beta(imputer, data, n_jobs, chunks=chunks)

    # Convert to IntervalFrame
    data = IntervalFrame(intervals=imputer.features, df = data.T)
    
    return data