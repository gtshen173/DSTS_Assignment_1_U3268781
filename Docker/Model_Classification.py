
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import Preprocessing as py


def classfication():
    zomato_df = py.get_data()
    zomato_df = py.preprocessing(zomato_df)
    zomato_df_encoded = py.one_hot_encode(zomato_df)
    zomato_df_encoded1 = py.set_target(zomato_df_encoded)
    
    X_train_selected, X_test_selected, y_train, y_test = py.train_test_split_class(zomato_df_encoded1)
    model_df = classfication_model(X_train_selected, X_test_selected, y_train, y_test)
    return model_df
    



def classfication_model(X_train_selected, X_test_selected, y_train, y_test):
    

    model_classification_3 = LogisticRegression(random_state=0)
    model_classification_3.fit(X_train_selected, y_train)
    y_pred_lr = model_classification_3.predict(X_test_selected)

    # Assuming y_test and y_pred are already available
    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)



    # Fitting the Random Forest model
    model_classification_4 = RandomForestClassifier(random_state=0)
    model_classification_4.fit(X_train_selected, y_train)
    y_pred_rf = model_classification_4.predict(X_test_selected)

    # Calculating the confusion matrix and accuracy
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    # Fitting the SVC model
    model_svc = SVC(random_state=0)
    model_svc.fit(X_train_selected, y_train)
    y_pred_svc = model_svc.predict(X_test_selected)

    # Calculating confusion matrix and accuracy
    conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)

    # Fitting the K-Nearest Neighbors (KNN) model
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train_selected, y_train)
    y_pred_knn = model_knn.predict(X_test_selected)

    # Calculating confusion matrix and accuracy for KNN
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    # Creating a DataFrame to store the accuracy and confusion matrix for each model
    # Creating a DataFrame to store the accuracy and confusion matrix for each model
    model_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'SVC', 'KNN'],
        'Accuracy': [accuracy_lr, accuracy_rf, accuracy_svc, accuracy_knn],
        'Confusion Matrix': [
            conf_matrix_lr,  # 2D format
            conf_matrix_rf,  # 2D format
            conf_matrix_svc, # 2D format
            conf_matrix_knn  # 2D format
        ]
    }


    # Converting the data to a pandas DataFrame
    model_df = pd.DataFrame(model_data)

    return model_df