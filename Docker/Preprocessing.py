import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso




def get_data():
    # Load the data
    csv_file = 'data/zomato_df_final_data.csv'
    # Reading the Zomato data and GeoJSON data
    
    zomato_df = pd.read_csv(csv_file)
    return zomato_df
    


def preprocessing(zomato_df):
    # Check for duplicates in all hashable columns (excluding columns that may contain lists)
    hashable_columns = [col for col in zomato_df.columns if zomato_df[col].apply(type).isin([list]).sum() == 0]

    # Check duplicates and drop them
    zomato_df.duplicated(subset=hashable_columns).sum()
    zomato_df.drop_duplicates(subset=hashable_columns, inplace=True)
    # Impute missing 'cost' with median
    zomato_df['cost'].fillna(zomato_df['cost'].median(), inplace=True)

    zomato_df['cost_2'].fillna(zomato_df['cost_2'].median(), inplace=True)

    # Impute missing 'lat' and 'lng' with mean values
    zomato_df['lat'].fillna(zomato_df['lat'].mean(), inplace=True)
    zomato_df['lng'].fillna(zomato_df['lng'].mean(), inplace=True)

    # Drop rows where 'rating_number', 'rating_text', or 'votes' are missing
    zomato_df.dropna(subset=['rating_number', 'rating_text', 'votes'], inplace=True)
    # Convert 'groupon' to a binary variable (True/False -> 1/0)
    zomato_df['groupon'] = zomato_df['groupon'].astype(int)

    return(zomato_df)

def one_hot_encode(zomato_df):
    # Convert lists to strings for 'subzone', 'type', and 'cuisine' 
    zomato_df['cuisine'] = zomato_df['cuisine'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    zomato_df['subzone'] = zomato_df['subzone'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    zomato_df['type'] = zomato_df['type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Apply one-hot encoding to 'subzone', 'type', and 'cuisine'
    zomato_df_encoded = pd.get_dummies(zomato_df, columns=['subzone', 'type', 'cuisine'], prefix=['subzone', 'type', 'cuisine'])

    return zomato_df_encoded


def featrue_selection_regression(zomato_df_encoded):
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
    
    return zomato_df_cleaned

# Step 2: Simplify the 'rating_text' column into two classes
def simplify_ratings(rating):
    if rating in ['Poor', 'Average']:
        return 1  # Class 1
    elif rating in ['Good', 'Very Good', 'Excellent']:
        return 2  # Class 2
    else:
        return None  # Exclude other records



def set_target(zomato_df_encoded):
    # Apply simplification to create binary classification
    zomato_df_encoded['binary_rating'] = zomato_df_encoded['rating_text'].apply(simplify_ratings)

    # Step 3: Drop rows with no classification in 'binary_rating'
    zomato_df_encoded1 = zomato_df_encoded.dropna(subset=['binary_rating'])
    return zomato_df_encoded1


def train_test_split_reg(zomato_df_cleaned):
    # Separate the features (X) and the target (y)
    X = zomato_df_cleaned.drop(columns=['rating_number'])  # Exclude the 'rating_number' column from the features
    y = zomato_df_cleaned['rating_number']  # Set 'rating_number' as the target variable

    X_scaled = scaling(X)
    
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    X_train_selected, X_test_selected = dim_reduce_lasso(X_train,X_test, y_train,X)
    
    return X_train_selected, X_test_selected, y_train, y_test


def scaling(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

 
def dim_reduce_lasso(X_train, X_test, y_train,X):\
    # Initialize Lasso model with a chosen alpha (regularization strength)
    lasso = Lasso(alpha=0.01)  # You can adjust alpha as needed

    # Fit the model on the training data
    lasso.fit(X_train, y_train)

    # Identify the important features (those with non-zero coefficients)
    important_features = [feature for feature, coef in zip(X.columns, lasso.coef_) if coef != 0]

    # Reduce X to only important features
    X_train_selected = X_train[:, lasso.coef_ != 0]
    X_test_selected = X_test[:, lasso.coef_ != 0]
    
    return X_train_selected, X_test_selected
     
    
     
    
 
 
def train_test_split_class(zomato_df_encoded1):
    # Select features (we will drop non-numeric columns and the target column)
    X = zomato_df_encoded1.drop(columns=['rating_text', 'binary_rating','rating_number', 'address', 'link', 'phone', 'title',  'color', 'cuisine_color'])
    y = zomato_df_encoded1['binary_rating']
    
    X_scaled = scaling(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    X_train_selected, X_test_selected = dim_reduce_lasso(X_train,X_test, y_train, X)
    
    return X_train_selected, X_test_selected, y_train, y_test

    

    
    
       













