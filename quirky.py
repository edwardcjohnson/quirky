import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from skleanr.preprocessing import OrdinalEncoder
import shap
from xgboost import XGBregressor
import lightgbm as lgbm

class Quirky(object):
    """
    Usage of the Quirky class:
    quirky = Quirky(random_seed=1)
    quirky.fit(df[X])
    df_explainer = quirky.explain_anomaly(df_test[X])
    j = 0
    shap.force_plot(quirky.explainer.expected_value, quirky.explainer_values[j,:], df_test[X].iloc[j])
    """

    def __init__(self, random_seed):
        """Return a Quirky object"""
        self.random_seed = random_seed
    
    def ordinal_encode_features(self, df, features, save_encoder_obj=False,
        encoder_obj_loc='./'):
        """Ordinal encode features for design matrix compatibility.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features
        features : List of categorical feature names
        save_encoder_obj : Boolean describing whether to save 
        the encoder object
        encoder_obj_loc : string describing the directory where
        encoder object should be saved.
        

        Returns
        -------
        df : Pandas dataframe with features encoded
        
        """
        
        encoder = OrdinalEncoder()
        encoder.fit(df[features])
        df[features] = encoder.transform(df[features])
        
        if save_encoder_obj:
            joblib.dump(encoder, '{}ordinal_encoder'.format(encoder_obj_loc))
        return df
    
    
    def fit_iso_forest(self, df):
        """
        Fit the isolation forest model to obtain anomaly scores.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features 

        Returns
        -------
        self : Object
        """
        random_state = np.random.RandomState(self.random_seed)
        self.iso_forest = IsolationForest(
            n_estimators = 1000,
            max_features = 0.8,
            behaviour = 'new',
            max_samples = 250, # 'auto':= max_samples = min(256, n_samples)
            random_state = random_state,
            contamination = 'auto'
            )
        self.iso_forest.fit(df)
        return self

    def fit_xgboost(self, df, y):
        """
        Fit the xgboost gbm model using the isolation forest's scores.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features
        y : array-like, shape = [n_samples]

        Returns
        -------
        self : Object
        """

        self.xgb = XGBRegressor(
            max_depth = 10,
            learning_rate = 0.01,
            n_estimators = 1000,
            tree_method = 'hist', #'exact'
            seed = self.random_seed
        )
        self.xgb.fit(df.values, self.iso_forest_preds)
        return self
    
    def fit_lightgbm(self, df, y, X_cat):
        """
        Fit the lightgbm model using the isolation forest's scores
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features
        y : array-like, shape = [n_samples]
        X_cat : list of feature names in df


        Returns
        -------
        self : Object
        """

        self.gbm = lgbm.LGBMRegressor(
            objective = 'regression',
            n_jobs = 4,
            num_leaves = 63,
            learning_rate = 0.01,
            n_estimators = 1000
        )
        df[X_cat] =  df[X_cat].astype('category')
        self.gbm.fit(df, self.iso_forest_preds,
        categorical_features = X_cat)
        return self

    def fit(self, df, save_model_obj=False, model_obj_loc='./'):
        """
        Fit the anomaly interpretion model using the isolation forest and gbm.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features

        Returns
        -------
        self : Object
        """
        self.fit_iso_forest(df)
        self.iso_forest_preds = self.iso_forest.decision_function(df) #lower -> more abnormal
        self.fit_xgboost(df, self.iso_forest_preds)
        # self.fit_lightgbm(df, self.iso_forest_preds, X_cat)
        if save_model_obj:
            joblib.dump(self.iso_forest,
                '{}iso_forest_object.joblib'.format(model_obj_loc))
            joblib.dump(self.xgb,
                '{}xgboost_object.joblib'.format(model_obj_loc))
        return self

    def predict(self, df):
        """
        Predict the GBM based anomaly score 
        for the anomaly interpretion model.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features

        Returns
        -------
        y : array-like, shape = [n_samples]
        """
        return self.gbm.predict(df.values)

    def explain_anomaly(self, df):
        """
        Generate the explainer values for each feature in the
        anomaly interpretion model.
        
        Parameters
        ----------
        df : Pandas dataframe where the columns are features

        Returns
        -------
        df_explainer : Pandas dataframe, shape = [n_samples, n_columns]
        """

        #df[X_cat] = df[X_cat].astype('category)
        self.explainer = shap.TreeExplainer(self.xgb)
        self.explainer_values = self.explainer.shap_values(df) #tree_limit=2,approximate=True)
        return pd.DataFrame(self.explainer_values, columns = df.columns)

    def load_model(self, model_obj_loc='./'):
        self.iso_forest = joblib.load(
            '{}iso_forest_object.joblib'.format(model_obj_loc)
        )
        self.xgb = joblib.load('{}xgboost_object.joblib'.format(model_obj_loc))
        return self
