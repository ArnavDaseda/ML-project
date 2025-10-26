import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        # Iterate over the models dictionary using .items()
        for model_name, model_obj in models.items():
            # Get the parameter grid for the current model using its name
            para = param[model_name]

            gs = GridSearchCV(model_obj, para, cv=3)
            gs.fit(x_train, y_train)

            model_obj.set_params(**gs.best_params_)
            model_obj.fit(x_train, y_train)

            y_train_pred = model_obj.predict(x_train)
            y_test_pred = model_obj.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the results with the model's name
            report[model_name] = (test_model_score, gs.best_params_)  

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as load_obj:
            return dill.load(load_obj)
    except Exception as e:
        raise CustomException(e,sys)