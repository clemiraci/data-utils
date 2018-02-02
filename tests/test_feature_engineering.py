import unittest
import numpy as np
import pandas as pd
from data_utils.feature_engineering import FeatureEngineering


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = pd.DataFrame()
        df['col_1'] = [np.random.random() for _ in range(10)]
        df['col_2'] = [np.random.random() for _ in range(10)]
        df['col_3'] = [np.random.random() for _ in range(10)]
        df['col_4'] = [np.random.random() for _ in range(10)]
        df['col_5'] = ['hello_%s'%i for i in range(10)]
        features = ['col_1', 'col_2', 'col_3', 'col_1^2', 'col_1*col_2', 'col_1/col_2', 'col_5', 'col_5*col_1', 'col_6']
        cls.feature_eng = FeatureEngineering(df, features)

    def test_combined_columns(self):
        dummy_df, features = self.feature_eng.create_dummy_variables()
        X, y = self.feature_eng.crate_feature_matrix(dummy_df, features, 'col_4')
        self.assertEqual((self.df['col_1'] * self.df['col_1']), self.df['col_1^2'])
        self.assertEqual((self.df['col_1'] * self.df['col_2']), self.df['col_1*col_2'])
        self.assertEqual((self.df['col_1'] / self.df['col_1']), self.df['col_1/col_2'])


if __name__ == "__main__":
    unittest.main()


#df.set_value(5, 'col_5', 'ciao')
#df.set_value(6, 'col_5', 'ciao_ciao')
#df['col_6'] = ['hola' for _ in range(10)]
#df.set_value(5, 'col_6', 'salut')


#features = ['col_1', 'col_2', 'col_3', 'col_1^2', 'col_1*col_2', 'col_1/col_2', 'col_5', 'col_5*col_1', 'col_6']
#feature_eng = FeatureEngineering(df, features)

#dummy_df, features = feature_eng.create_dummy_variables()
#X,y = feature_eng.crate_feature_matrix(dummy_df, features, 'col_4')

#(df['col_1'] * df['col_1'] == df['col_1^2']).all()
#(df['col_1'] * df['col_2'] == df['col_1*col_2']).all()
#(df['col_1'] / df['col_2'] == df['col_1/col_2']).all()
