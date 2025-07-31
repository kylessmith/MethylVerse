import numpy as np
import pandas as pd
import json
from scipy.special import softmax
from ..Architectures.MultiClassEnsemble import EnsembleModel
import torch


def logsoftmax(x):
    c = np.max(x, axis=1, keepdims=True)  # Max per row for numerical stability
    logsumexp = np.log(np.sum(np.exp(x - c), axis=1, keepdims=True))  # Log-sum-exp per row
    return x - c - logsumexp  # Log-softmax formula


class MPACT_classifier(object):
    """
    """
    
    def __init__(self,
                 directory,
                 threshold = 0.7):
        """
        """

        self.device = torch.device("cpu")
        self.models = [EnsembleModel.load(directory+"MPACT_D1_v1.pth", device=self.device),
                       EnsembleModel.load(directory+"MPACT_S1_v1.pth", device=self.device),
                       EnsembleModel.load(directory+"MPACT_S2_v1.pth", device=self.device)]
        self.features1 = np.loadtxt(directory+"Features1_v4.txt", dtype=str)
        self.features2 = np.loadtxt(directory+"Features2_v4.txt", dtype=str)
        self.features3 = np.loadtxt(directory+"Features3_v4.txt", dtype=str)

        self.codings = json.load(open(directory+"codings.json"))
        self.class_merge = self.codings["merge"]
        self.decoder = self.codings["decoder"]

        self.threshold = threshold
    

    def predict_model(self,
                      input1,
                      input2,
                      input3,
                      model_number,
                      merge = True):
        """
        """
        # Load model
        model = self.models[model_number - 1]

        # Check input shapes
        dim = 2
        if input1.shape[0] == 1:
            dim = 1
            # Add dummy rows for single sample input
            input1 = pd.DataFrame(np.repeat(input1, 2, axis=0), 
                                  index=[input1.index.values[0], input1.index.values[0]],
                                  columns=input1.columns)
            input2 = pd.DataFrame(np.repeat(input2, 2, axis=0), 
                                  index=[input2.index.values[0], input2.index.values[0]],
                                  columns=input2.columns)
            input3 = pd.DataFrame(np.repeat(input3, 2, axis=0), 
                                  index=[input3.index.values[0], input3.index.values[0]],
                                  columns=input3.columns)

        probabilities, predictions = model.predict_step({"x": torch.from_numpy(input1.values.astype(np.float32))},
                                                        {"x": torch.from_numpy(input2.values.astype(np.float32))},
                                                        {"x": torch.from_numpy(input3.values.astype(np.float32))})
        probabilities = probabilities.detach().cpu().numpy()

        # Reshape probabilities if single sample input
        if dim == 1:
            probabilities = probabilities[[0],:]
            predictions = predictions[[0]]
            input1 = input1.iloc[[0],:]
            input2 = input2.iloc[[0],:]
            input3 = input3.iloc[[0],:]

        # Get classes
        classes = pd.Series(self.decoder).values

        # Merge classes
        probabilities = pd.DataFrame(probabilities, index=input1.index.values, columns=classes)
        if merge:
            for c in self.class_merge:
                probabilities.loc[:,c] = probabilities.loc[:,self.class_merge[c]].sum(axis=1).values
                probabilities = probabilities.drop(columns=list(set(self.class_merge[c]) - set([c])))
        
        # Determine predictions
        predictions = probabilities.columns.values[probabilities.values.argmax(axis=1)]
        
        return predictions, probabilities
        
        
    def predict(self,
                X: pd.DataFrame,
                method: str = "max",
                transform: bool = True):
        """
        """

        # Scale X
        if transform:
            X = (X * 2) - 1  # Scale to [-1, 1] for MLP input

        # Divide X into feature folds
        input1 = X.loc[:,self.features1]
        input2 = X.loc[:,self.features2]
        input3 = X.loc[:,self.features3]
        
        # Get dense predictions
        dense_predictions, dense_probabilities = self.predict_model(input1, input2, input3, 1, merge=True)
        
        # Get sparse model 1 predictions
        sparse1_predictions, sparse1_probabilities = self.predict_model(input1, input2, input3, 2, merge=True)
        
        # Get sparse model 2 predictions
        sparse2_predictions, sparse2_probabilities = self.predict_model(input1, input2, input3, 3, merge=True)

        # Determine probabilities
        if method == "average":
            probabilities = (dense_probabilities + sparse1_probabilities + sparse2_probabilities) / 3
        elif method == "max":
            probabilities = dense_probabilities.copy()
            # Iterate over sparse models and take max probabilities
            for i in range(sparse1_probabilities.shape[0]):
                if sparse1_probabilities.values[i,:].max() > probabilities.values[i,:].max():
                    probabilities.iloc[i,:] = sparse1_probabilities.values[i,:]
                if sparse2_probabilities.values[i,:].max() > probabilities.values[i,:].max():
                    probabilities.iloc[i,:] = sparse2_probabilities.values[i,:]
        elif method == "concat":
            probabilities = pd.concat([dense_probabilities, sparse1_probabilities, sparse2_probabilities]).groupby(level=0).max()
            # Sum to 1
            probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
        else:
            raise ValueError("Method not recognized.")
        
        # Determine predictions
        probabilities = probabilities.loc[X.index.values,:]
        predictions = probabilities.columns.values[probabilities.values.argmax(axis=1)]

        return predictions, probabilities