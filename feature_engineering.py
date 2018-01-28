import numpy as np
import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe, features):
        self.df = dataframe
        self.features = features

    def _feature_classification(self, feature_name):
        feature_type = 'normal'
        if '*' in feature_name:
            feature_type = 'product'
        elif '/' in feature_name:
            feature_type = 'ratio'
        elif '^' in feature_name:
            feature_type = 'power'
        return feature_type

    def _detect_dummy(self, feature_name):
        is_dummy = False
        if type(self.df[feature_name].iloc[0]) == str:
            is_dummy = True
        return is_dummy

    def _extract_information(self, feature_name, feature_type):
        value = np.nan
        if feature_type == 'power':
            value = feature_name.split('^')
            value[1] = np.float64(value[1])
        elif feature_type == 'product':
            value = feature_name.split('*')
        elif feature_type == 'ratio':
            value = feature_name.split('/')
        return value

    def create_dummy_variables(self):
        """
        dummy categorical columns need to be listed before their interaction with other columns
        :return:
        """
        def update_feature_list(dummy_name, dummy_values, all_features):
            complete_features = all_features[:]
            for f in all_features:
                if f == dummy_name:
                    complete_features = complete_features + dummy_values
                    complete_features.remove(f)
                elif f != dummy_name and dummy_name in f:
                    dummy_features = [f.replace(dummy_name, d) for d in dummy_values]
                    complete_features = complete_features + dummy_features
                    complete_features.remove(f)
            return complete_features

        new_features = self.features[:]
        df = self.df.copy()
        dummy_candidates = list(set(self.df.columns) & set(self.features))
        for f in dummy_candidates:
            if self._detect_dummy(f):
                dummy = pd.get_dummies(self.df[f], drop_first=True)
                df = pd.merge(df, dummy, left_index=True, right_index=True)
                new_features = update_feature_list(f, dummy.columns.tolist(), new_features)
        return df, new_features

    def _generate_combined_features(self, df, features_list):
        for f in features_list:
            feature_type = self._feature_classification(f)
            value = self._extract_information(f, feature_type)
            if feature_type == 'power':
                df[f] = df[value[0]]**value[1]
            elif feature_type == 'product':
                df[f] = df[value[0]] * df[value[1]]
            elif feature_type == 'ratio':
                df[f] = df[value[0]] / df[value[1]]
        return df

    def crate_feature_matrix(self, df, features, target):
        complete_df = self._generate_combined_features(df, features)
        X = complete_df[features].copy()
        y = complete_df[target]
        return X, y

    def detect_dummy_variables(self, feature_list):
        dummy_var = {}
        for f in feature_list:
            if self._detect_dummy(f):
                dummy_var[f] = self.df[f].unique().tolist()
        return dummy_var

