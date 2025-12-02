from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CombineAttributesAdder (BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.households_ix = 6
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
class MostImportantFeatureSelector (BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, num_features):
        self.feature_importances = feature_importances
        self.num_features = num_features
    
    def fit(self, X, y=None):
        self.feature_indices_ = np.argsort(self.feature_importances)[-self.num_features:]
        return self
    
    def transform(self, X, y=None):
        return X[:, self.feature_indices_]