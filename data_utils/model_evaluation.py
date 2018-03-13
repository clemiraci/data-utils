from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns

class evaluateRegression:
    def __init__(self, model, feature_matrix):
        self.model = model
        self.X = feature_matrix
        self.y_pred = model.predict(self.X)
        
    def get_scores(self, y_true):
        mae = mean_absolute_error(y_true, self.y_pred)
        mse = mean_squared_error(y_true, self.y_pred)
        mape = np.mean((self.y_pred - y_test) / y_test)
        return mae, mse, mape
    
    def plot_error_ristribution(self, y_true):
        error = self.y_pred - y_true
        plt.figure(figsize=(10,5))
        sns.distplot(error, kde=False, norm_hist=True)
        plt.title('Error distribution')
        plt.show()
        
    def plot_target_distributions(self, y_true, x_lim_1, x_lim_2):
        plt.figure(figsize=(10,5))
        sns.distplot(y_true, kde=False, norm_hist=True, label='True')
        sns.distplot(self.y_pred, kde=False, norm_hist=True, label='Prediction')
        plt.title('Observed vs Predicted')
        plt.legend()
        plt.xlim(x_lim_1, x_lim_2)
        plt.show()
        