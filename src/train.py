import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV 
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

import mlflow

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/nonsindisomkhize8/machinelearningpipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'nonsindisomkhize8'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '569a0634b5b1556e4eb2464bd05e8e95b6137830'

#Load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def random_tuning(X_train, y_train, search_space):
    rf = RandomForestClassifier()
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions = search_space,
        n_iter= 20,      
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    return random_search

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/nonsindisomkhize8/machinelearningpipeline.mlflow")

    #MLFLOW run
    with mlflow.start_run():
        # #split the dataset into training and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature=infer_signature(X_train,y_train)

        #Define search space
        search_space = {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt','log2']
        }
        
        #Perform hyperparameter tuning
        search_space=random_tuning(X_train,y_train,search_space)

        #get best model
        best_model=search_space.best_estimator_

        #predict and evaluate the model
        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        ## Log additional metrics 
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimatios",search_space.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", search_space.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", search_space.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", search_space.best_params_['min_samples_leaf'])

        #log the confusion matrix and classification report
        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        #create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])








