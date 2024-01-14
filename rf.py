import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        ...
        models = []
        self.train_idxs, self.test_idxs = [], []
        for est_id in range(self.n_estimators):
            train_ids =  resample(range(X.shape[0]), n_samples=int(2/3*X.shape[0]), replace=True, random_state=est_id)
            
            oob_ids = [j for j in range(len(X)) if j not in train_ids]

            self.train_idxs.append(train_ids)
            self.test_idxs.append(oob_ids)

            self.trees[est_id].fit(X[train_ids], y[train_ids])
            models.append(self.trees[est_id])

            self.models = models
        
        if self.oob_score:
            self.oob_score_ = self.calculate_oob_scores(X, y, self.test_idxs)
            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [RegressionTree621(min_samples_leaf, max_features) for i in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        weighted_avg = np.zeros_like(X_test[:, 0])
        weighted_sum = np.zeros_like(X_test[:, 0])

        for dtree_i in self.trees:
            predicted_leafs = dtree_i.predict(X_test)

            for i in range(len(predicted_leafs)):
                leaf_samples, leaf_predictions = predicted_leafs[i].n, predicted_leafs[i].prediction

                weighted_avg[i]+=leaf_samples*leaf_predictions
                weighted_sum[i]+=leaf_samples
        
        return weighted_avg/weighted_sum

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))

    '''def calculate_oob_scores(self, X, y, oob_indexes):

        oob_predictions_array = [[]] * len(X)

        for est_id in range(self.n_estimators):

            leaf_nodes_list = self.models[est_id].predict(X[oob_indexes[est_id]])
            predictions = [x.prediction for x in leaf_nodes_list]

            for i in range(len(predictions)):
                oob_predictions_array[i].append(predictions[i])

            for i, est_id in enumerate(range(len(oob_indexes))):
                for idx, val in zip(oob_indexes[est_id], predictions):
                    oob_predictions_array[idx].append(val)


        oob_means = [np.mean(sublist) for sublist in oob_predictions_array]

        nan_indices = [i for i, mode_val in enumerate(oob_means) if np.isnan(mode_val)]
        oob_means_filtered = [mode_val for i, mode_val in enumerate(oob_means) if i not in nan_indices]
        y_fil = [y[i] for i in range(len(y)) if i not in nan_indices]

        return r2_score(y_fil, oob_means_filtered)'''
    
    #Expanded the list comprehensions and tried to write it in simple format
    def calculate_oob_scores(self, X, y, oob_indexes):
        oob_predictions_array = [[] for i in range(X.shape[0])]

        for est_id in range(self.n_estimators):

            leaf_nodes_list = self.models[est_id].predict(X[oob_indexes[est_id]])

            predictions = []
            for x in leaf_nodes_list:
                predictions.append(x.prediction)
            
            for i in range(len(oob_indexes[est_id])):
                oob_predictions_array[oob_indexes[est_id][i]].append(predictions[i])

        oob_means = [np.mean(sublist) for sublist in oob_predictions_array]

        nan_indices = [i for i, mode_val in enumerate(oob_means) if np.isnan(mode_val)]
        oob_means_filtered = [mode_val for i, mode_val in enumerate(oob_means) if i not in nan_indices]
        y_fil = [y[i] for i in range(len(y)) if i not in nan_indices]

        return r2_score(y_fil, oob_means_filtered)
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = [ClassifierTree621(min_samples_leaf, max_features) for i in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        '''class_freqencies = []
        for tree in self.trees:
            leaf_predictions = [leaf.prediction for leaf in tree.predict(X_test)]
            class_freqencies.append(leaf_predictions)

        return stats.mode(class_freqencies, axis = 1).mode.flatten()'''

        #Expanded the previous function and changed the type into numpy array
        class_freqencies = []
        for tree in self.trees:
            leaf_predictions = [leaf.prediction for leaf in tree.predict(X_test)]
            class_freqencies.append(leaf_predictions)
        class_freqencies = np.array(class_freqencies).T
        return stats.mode(class_freqencies, axis=1).mode.flatten()



    def predict(self, X_test) -> np.ndarray:
        class_counts = np.zeros((X_test.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees):
            leaf_nodes = tree.predict(X_test)

            for j, leaf in enumerate(leaf_nodes):
                class_counts[j][i] = leaf.prediction

        mode_predictions = stats.mode(
            class_counts,
            axis=1,
            keepdims=False,
        )

        return mode_predictions.mode.flatten()

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test,self.predict(X_test))
    
    '''def calculate_oob_scores(self, X, y, oob_indexes):

        oob_predictions_array = [[]] * len(X)

        for est_id in range(self.n_estimators):

            leaf_nodes_list = self.models[est_id].predict(X[oob_indexes[est_id]])
            predictions = [x.prediction for x in leaf_nodes_list]

            for i in range(len(predictions)):
                oob_predictions_array[i].append(predictions[i])

            for i, est_id in enumerate(range(len(oob_indexes))):
                for idx, val in zip(oob_indexes[est_id], predictions):
                    oob_predictions_array[idx].append(val)


        oob_modes = [stats.mode(sublist)[0] for sublist in oob_predictions_array]

        nan_indices = [i for i, mode_val in enumerate(oob_modes) if np.isnan(mode_val)]
        oob_modes_filtered = [mode_val for i, mode_val in enumerate(oob_modes) if i not in nan_indices]
        y_fil = [y[i] for i in range(len(y)) if i not in nan_indices]
        print(len(y_fil), len(oob_modes_filtered))
        return accuracy_score(y_fil, oob_modes_filtered)'''
    

    #Expanded the list comprehensions and tried to write it in simple format
    def calculate_oob_scores(self, X, y, oob_indexes):
        oob_predictions_array = [[] for i in range(X.shape[0])]

        for est_id in range(self.n_estimators):

            leaf_nodes_list = self.models[est_id].predict(X[oob_indexes[est_id]])

            predictions = []
            for x in leaf_nodes_list:
                predictions.append(x.prediction)
            
            for i in range(len(oob_indexes[est_id])):
                oob_predictions_array[oob_indexes[est_id][i]].append(predictions[i])


        oob_modes = [stats.mode(sublist)[0] for sublist in oob_predictions_array]

        nan_indices = [i for i, mode_val in enumerate(oob_modes) if np.isnan(mode_val)]
        oob_modes_filtered = [mode_val for i, mode_val in enumerate(oob_modes) if i not in nan_indices]
        y_fil = [y[i] for i in range(len(y)) if i not in nan_indices]
        print(len(y_fil), len(oob_modes_filtered))
        return accuracy_score(y_fil, oob_modes_filtered)
        