import pandas as pd
import geopandas as gpd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score


# Load the data
csv_file = 'data/zomato_df_final_data.csv'


# Reading the Zomato data and GeoJSON data
zomato_df = pd.read_csv(csv_file)





zomato_df.info()

#Zomato dataframe have 10500 values(rows) with 17features(columns)
zomato_df.shape

zomato_df.head()

zomato_df.columns.tolist()



# Identify categorical columns (object or categorical types)
categorical_columns = zomato_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Identify numerical columns (integer and float types)
numerical_columns = zomato_df.select_dtypes(include=['number']).columns.tolist()

print("Categorical Columns \n", categorical_columns)

print("Numerical Columns \n",numerical_columns)

# Check for duplicates in all hashable columns (excluding columns that may contain lists)
hashable_columns = [col for col in zomato_df.columns if zomato_df[col].apply(type).isin([list]).sum() == 0]

# Check duplicates and drop them
zomato_df.duplicated(subset=hashable_columns).sum()
zomato_df.drop_duplicates(subset=hashable_columns, inplace=True)


# Checking for missing values in the dataset
missing_values = zomato_df.isnull().sum()

# Display the number of missing values per column
missing_values

# Impute missing 'cost' with median
zomato_df['cost'].fillna(zomato_df['cost'].median(), inplace=True)

zomato_df['cost_2'].fillna(zomato_df['cost_2'].median(), inplace=True)

# Impute missing 'lat' and 'lng' with mean values
zomato_df['lat'].fillna(zomato_df['lat'].mean(), inplace=True)
zomato_df['lng'].fillna(zomato_df['lng'].mean(), inplace=True)

# Drop rows where 'rating_number', 'rating_text', or 'votes' are missing
zomato_df.dropna(subset=['rating_number', 'rating_text', 'votes'], inplace=True)

# Verify the cleaning process
cleaned_missing_values = zomato_df.isnull().sum()

# Display the cleaned missing values
cleaned_missing_values


# Convert lists to strings for 'subzone', 'type', and 'cuisine' 
zomato_df['cuisine'] = zomato_df['cuisine'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
zomato_df['subzone'] = zomato_df['subzone'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
zomato_df['type'] = zomato_df['type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Apply one-hot encoding to 'subzone', 'type', and 'cuisine'
zomato_df_encoded = pd.get_dummies(zomato_df, columns=['subzone', 'type', 'cuisine'], prefix=['subzone', 'type', 'cuisine'])

# Convert 'groupon' to a binary variable (True/False -> 1/0)
zomato_df_encoded['groupon'] = zomato_df_encoded['groupon'].astype(int)

# Verify the one-hot encoding
zomato_df_encoded.head()


zomato_df_encoded.shape, zomato_df_encoded.columns.tolist()

# List of columns that are not needed for the model
columns_to_remove = ['address', 'cuisine', 'link', 'phone', 'title', 'subzone', 'type', 'rating_text', 'color', 'cuisine_color']

# Remove the unnecessary columns from the dataset
zomato_df_cleaned = zomato_df_encoded.drop(columns=columns_to_remove, errors='ignore')

# Additionally, if there are columns with too many missing values, remove those as well
# For example, if you want to remove columns with more than 30% missing values:
threshold = 0.3 * zomato_df_encoded.shape[0]
columns_with_many_missing = zomato_df_encoded.isnull().sum()[zomato_df_encoded.isnull().sum() > threshold].index.tolist()

# Drop columns with too many missing values
zomato_df_cleaned = zomato_df_cleaned.drop(columns=columns_with_many_missing, errors='ignore')

# Display the cleaned dataset
zomato_df_cleaned.shape


# Separate the features (X) and the target (y)
X = zomato_df_cleaned.drop(columns=['rating_number'])  # Exclude the 'rating_number' column from the features
y = zomato_df_cleaned['rating_number']  # Set 'rating_number' as the target variable

X.shape, y.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)



# Initialize Lasso model with a chosen alpha (regularization strength)
lasso = Lasso(alpha=0.01)  # You can adjust alpha as needed

# Fit the model on the training data
lasso.fit(X_train, y_train)

# Identify the important features (those with non-zero coefficients)
important_features = [feature for feature, coef in zip(X.columns, lasso.coef_) if coef != 0]

# Reduce X to only important features
X_train_selected = X_train[:, lasso.coef_ != 0]
X_test_selected = X_test[:, lasso.coef_ != 0]

print(f"Number of selected features: {len(important_features)}")
print(f"Selected features: {important_features}")





print(f"Mean Squared Error (MSE) for Model 1: {mse_lr}")




print(f"Mean Squared Error (MSE) for Gradient Descent Linear Regression model: {mse_gd}")

from sklearn.linear_model import SGDRegressor

print(f"Mean Squared Error (MSE) for SGDRegressor (Gradient Descent) model: {mse_sgd}")












print(f"Number of selected features: {len(important_features)}")
print(f"Selected features: {important_features}")


model_classification_3 = LogisticRegression(random_state=0)
model_classification_3.fit(X_train_selected, y_train)
y_pred_lr = model_classification_3.predict(X_test_selected)

# Assuming y_test and y_pred are already available
conf_matrix = confusion_matrix(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})

# Set plot titles and labels
plt.title(f'Confusion Matrix for Logistic Regression\nAccuracy: {accuracy_lr:.4f}', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Display the plot with accuracy on top
plt.show()

# Check the distribution of binary_rating
binary_rating_distribution = zomato_df_encoded1['binary_rating'].value_counts()
print(binary_rating_distribution)

# Plotting the distribution of binary_rating
binary_rating_distribution.plot(kind='bar')
plt.title('Distribution of Binary Rating')
plt.xlabel('Binary Rating')
plt.ylabel('Count')
plt.xticks([0, 1], ['0', '1'], rotation=0)
plt.show()



# Plotting the confusion matrix with accuracy
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix for Random Forest\nAccuracy: {accuracy_rf:.4f}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



# Plotting the confusion matrix with accuracy for SVC
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix for SVC\nAccuracy: {accuracy_svc:.4f}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Plotting the confusion matrix with accuracy for KNN
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix for KNN\nAccuracy: {accuracy_knn:.2f}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



