import onnxruntime as rt
import numpy as np
import pandas as pd
import json


def logsoftmax(x):
    c = np.max(x, axis=1, keepdims=True)  # Max per row for numerical stability
    logsumexp = np.log(np.sum(np.exp(x - c), axis=1, keepdims=True))  # Log-sum-exp per row
    return x - c - logsumexp  # Log-softmax formula


class DenseModel(object):
    """
    """
    
    def __init__(self,
                 directory):
        """
        """
        
        # Read models
        self.models = []
        self.models.append(rt.InferenceSession(directory+"MPACT_dense1_v1.onnx", providers=["CPUExecutionProvider"]))
        self.models.append(rt.InferenceSession(directory+"MPACT_dense2_v1.onnx", providers=["CPUExecutionProvider"]))
        
        # Read features
        self.features = []
        self.features.append(np.array([line.strip() for line in open(directory + "Features1_v4.txt","r")]))
        self.features.append(np.array([line.strip() for line in open(directory + "Features2_v4.txt","r")]))
        
        # Read merges
        self.class_merge = json.load(open(directory + "codings.json","r"))["merge"]
        
        # Store attributes
        self.input_name = self.models[0].get_inputs()[0].name

         
    def predict(self,
                X,
                merge = True):
        """
        """
        
        # Get classes
        classes = pd.Series(self.models[0].run(None, {self.input_name: np.ones((2,len(self.features[0]))).astype(np.float32)})[1][0]).index.values
        
        # Iterate over models
        probabilities = pd.DataFrame(np.zeros((X.shape[0], 97)), index=X.index.values, columns=classes)
        for i, model in enumerate(self.models):
            pred, model_probabilities = model.run(None, {self.input_name: X.loc[:,self.features[i]].values.astype(np.float32)})
            model_probabilities = pd.DataFrame(model_probabilities, index=X.index.values, columns=classes)

            # Check if model probabilities are higher
            for j in range(X.shape[0]):
                obs = X.index.values[j]
                if model_probabilities.loc[obs,:].max() > probabilities.loc[obs,:].max():
                    probabilities.loc[obs,:] = model_probabilities.loc[obs,:]
                
        # Merge classes
        if merge:
            for c in self.class_merge:
                probabilities.loc[:,c] = probabilities.loc[:,self.class_merge[c]].sum(axis=1).values
                probabilities = probabilities.drop(columns=list(set(self.class_merge[c]) - set([c])))
        
        # Determine predictions
        predictions = probabilities.columns.values[probabilities.values.argmax(axis=1)]
        
        return predictions, probabilities
        

class SparseModel(object):
    """
    """
    
    def __init__(self,
                 directory,
                 model_number = 1):
        """
        """
        
        # Get model number
        self.number = str(model_number)
        
        # Read models
        self.models = []
        self.models.append(rt.InferenceSession(directory+"MPACT_model"+self.number+"_1_v1.onnx", providers=["CPUExecutionProvider"]))
        self.models.append(rt.InferenceSession(directory+"MPACT_model"+self.number+"_2_v1.onnx", providers=["CPUExecutionProvider"]))
        self.models.append(rt.InferenceSession(directory+"MPACT_model"+self.number+"_3_v1.onnx", providers=["CPUExecutionProvider"]))
        
        # Read features
        self.features = []
        self.features.append(np.array([line.strip() for line in open(directory + "Features1_v4.txt","r")]))
        self.features.append(np.array([line.strip() for line in open(directory + "Features2_v4.txt","r")]))
        self.features.append(np.array([line.strip() for line in open(directory + "Features3_v4.txt","r")]))
        
        # Read merges
        self.class_merge = json.load(open(directory + "codings.json","r"))["merge"]
        self.decoder = json.load(open(directory + "codings.json","r"))["decoder"]
        
        # Store attributes
        self.input_name = self.models[0].get_inputs()[0].name
        self.weights = np.array([1.00, 0.5, 0.1]).astype(np.float32)
        
         
    def predict(self,
                X,
                merge = True):
        """
        """
        
        # Get classes
        classes = pd.Series(self.decoder).values
        
        # Iterate over models
        probabilities = pd.DataFrame(np.zeros((X.shape[0], len(classes))), index=X.index.values, columns=classes)
        for i, model in enumerate(self.models):
            pred = model.run(None, {self.input_name: X.loc[:, self.features[i]].values.astype(np.float32)})
            prob = pd.DataFrame(np.exp(logsoftmax(pred[0])), index=X.index.values, columns=classes)
            probabilities += prob * self.weights[i]
        
        # Determine weighted average
        probabilities /= self.weights.sum()
        
        # Merge classes
        if merge:
            for c in self.class_merge:
                probabilities.loc[:,c] = probabilities.loc[:,self.class_merge[c]].sum(axis=1).values
                probabilities = probabilities.drop(columns=list(set(self.class_merge[c]) - set([c])))
        
        # Determine predictions
        predictions = probabilities.columns.values[probabilities.values.argmax(axis=1)]
        
        return predictions, probabilities


class MPACT_classifier(object):
    """
    """
    
    def __init__(self,
                 directory,
                 threshold = 0.7):
        """
        """
        
        self.dense_model = DenseModel(directory)
        self.sparse_models = [SparseModel(directory, model_number=1),
                              SparseModel(directory, model_number=2)]
        self.threshold = threshold
        
        
    def predict(self,
                X,
                method = "max"):
        """
        """
        
        # Get dense predictions
        dense_predictions = self.dense_model.predict(X)[1]
        
        # Get sparse predictions
        sparse_predictions1 = self.sparse_models[0].predict(X)[1]
        sparse_predictions2 = self.sparse_models[1].predict(X)[1]
        sparse_predictions1 = sparse_predictions1.loc[dense_predictions.index.values,dense_predictions.columns]
        sparse_predictions2 = sparse_predictions2.loc[dense_predictions.index.values,dense_predictions.columns]

        # Determine probabilities
        if method == "average":
            probabilities = (dense_predictions + sparse_predictions1 + sparse_predictions2) / 3
        elif method == "max":
            probabilities = pd.concat([dense_predictions, sparse_predictions1, sparse_predictions2]).groupby(level=0).max()
        else:
            raise ValueError("Method not recognized.")
        
        # Determine predictions
        probabilities = probabilities.loc[X.index.values,:]
        predictions = probabilities.columns.values[probabilities.values.argmax(axis=1)]

        return predictions, probabilities

        