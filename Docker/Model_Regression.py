
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import Preprocessing as py
from sklearn.linear_model import LinearRegression


def regressor():
    zomato_df = py.get_data()
    zomato_df = py.preprocessing(zomato_df)
    zomato_df_encoded = py.one_hot_encode(zomato_df)
    zomato_df_cleaned = py.featrue_selection_regression(zomato_df_encoded)
    X_train_selected, X_test_selected, y_train, y_test = py.train_test_split_reg(zomato_df_cleaned)
    mse_df = regressor_model(X_train_selected, X_test_selected, y_train, y_test)
    return mse_df
    

def regressor_model(X_train_selected, X_test_selected, y_train, y_test):
    
    
    model_regression_1 = LinearRegression()
    model_regression_1.fit(X_train_selected, y_train)

    # Make predictions and evaluate the model
    y_pred = model_regression_1.predict(X_test_selected)
    from sklearn.metrics import mean_squared_error
    mse_lr = mean_squared_error(y_test, y_pred)


    # Create a class for the linear regression with Gradient Descent for multiple features
    class GradientDescentLinearRegression:
        def __init__(self, learning_rate, iterations=1000):
            self.learning_rate = learning_rate
            self.iterations = iterations
        
        def fit(self, X, y):
            # Ensure X and y are aligned
            X = pd.DataFrame(X).reset_index(drop=True)  # Ensure X is a pandas DataFrame and reset index
            y = pd.Series(y).reset_index(drop=True)     # Ensure y is a pandas Series and reset index
            
            # Initialize parameters (weights) for each feature plus the bias term
            n, m = X.shape  # n = number of samples, m = number of features
            self.theta = pd.Series(0, index=X.columns)  # Initialize weights to zero
            self.bias = 0.0  # Initialize bias (intercept) to zero
            
            # Gradient Descent Algorithm
            for _ in range(self.iterations):
                # Calculate predictions
                y_pred = X.dot(self.theta) + self.bias
                
                # Compute gradients
                d_theta = -(2 / n) * X.T.dot(y - y_pred)  # Gradient for weights (features)
                d_bias = -(2 / n) * (y - y_pred).sum()    # Gradient for the bias term
                
                # Update parameters
                self.theta = self.theta - self.learning_rate * d_theta
                self.bias = self.bias - self.learning_rate * d_bias
            
        def predict(self, X):
            X = pd.DataFrame(X).reset_index(drop=True)  # Ensure X is a pandas DataFrame and reset index
            return X.dot(self.theta) + self.bias


    # Example usage with multi-dimensional data
    model_regression_3 = GradientDescentLinearRegression(learning_rate=0.0021, iterations=1000)

    # Reset the indices of X_train_selected and y_train to ensure alignment
    X_train_selected = pd.DataFrame(X_train_selected).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)

    # Train the model
    model_regression_3.fit(X_train_selected, y_train)

    # Make predictions on the test set
    X_test_selected = pd.DataFrame(X_test_selected).reset_index(drop=True)
    y_pred_gd = model_regression_3.predict(X_test_selected)

    # Evaluate the model with Mean Squared Error (MSE)
    mse_gd = mean_squared_error(y_test, y_pred_gd)

    # Create and train the model using SGD (Gradient Descent)
    model_regression_2 = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.000021)
    model_regression_2.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred_sgd = model_regression_2.predict(X_test_selected)

    # Calculate Mean Squared Error (MSE)
    mse_sgd = mean_squared_error(y_test, y_pred_sgd)


    mse_data = {
        "Model": [ "Linear Regression (Model 1)", "SGDRegressor(Model 2)", "Batch Gradient Descent (Model 3)"],
        "MSE": [mse_lr, mse_sgd, mse_gd]
    }

    # Convert the dictionary into a pandas DataFrame
    mse_df = pd.DataFrame(mse_data)

    # Set the index to start from 1 and recreate the table
    mse_df.index = mse_df.index + 1

    # Print the table for visual confirmation
    return mse_df
