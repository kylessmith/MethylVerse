import pandas as pd
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from sknetwork.classification import DiffusionClassifier
from sknetwork.regression import Dirichlet
from projectframe import ProjectFrame
from joblib import Parallel, delayed
import os

# Local imports
from ...data.import_data import get_data_file
from ...core.utilities import to_microarray


class EPICimputer(object):
    """
    """
    
    def __init__(self, network_file="EPIC_probe_network.h5"):
        """
        """

        # Get File
        if not os.path.exists(network_file):
            network_file = get_data_file(network_file)
        
        # Read imputation matrix
        f = h5py.File(network_file, "r")
        self.features = np.array(f["names"]).astype(str)
        data = np.array(f["data"])
        indices = np.array(f["indices"])
        indptr = np.array(f["indptr"])
        shape = np.array(f["shape"])
        shape = (shape[0], shape[1])
        f.close()
        self.network = csr_matrix((data, indices, indptr), shape=shape)
        
        
    def match_matrix(self, X: pd.DataFrame):
        """
        """

        # Check if the features match
        if np.array_equal(self.features, X.columns.values):
            return X
        
        # Copy
        X_copy = X.copy()
        
        # Match features
        common = X_copy.columns.intersection(self.features)
        if len(common) == 0:
            common = X_copy.index.intersection(self.features)
            if len(common) == 0:
                raise ValueError("No matching features found.")
            X_copy = X_copy.T

        missing = pd.Index(self.features).difference(X_copy.columns.values)
        X_copy = X_copy.loc[:,common]
        #X_copy.loc[:,missing] = np.nan
        #for feature in missing:
        #    X_copy.loc[:,feature] = np.nan
        null_df = pd.DataFrame(np.nan, index=X_copy.index, columns=missing)
        X_copy = pd.concat([X_copy, null_df], axis=1)
        X_copy = X_copy.loc[:,self.features]
        
        return X_copy
        
    
    def impute_status(self, X, threshold=0.75):
        """
        """

        # Initialize the prediction matrix
        prediction = self.match_matrix(X)
    
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

        # Initialize the prediction matrix
        prediction = self.match_matrix(X)

        # Fit the Dirichlet model
        dirichlet = Dirichlet()

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

def impute_epic(data: pd.DataFrame,
                network_file: str = "EPIC_probe_network.h5",
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
    imputer = EPICimputer(network_file=network_file)
    
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
    
    return data