import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
     def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

     def initiate_model_trainer(self,train_array,test_array):
         try:
             x_train,y_train,x_test,y_test = (
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:,-1],
                )
             models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
             } 
             params={
                 "DecisionTree":{
                     'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                 },
                 "RandomForest":{
                     'n_estimators':[8,16,32,64,128],
                 },
                 "CatBoostRegressor":{
                     'depth':[6,8,10],
                     'learning_rate':[0.1,0.01,0.5,0.001],
                     'iterations':[10,30,50,100],
                 },
                 "GradientBoosting":{
                    'n_estimators':[8,16,32,64,128],
                    'learning_rate':[0.1,0.01,0.5,0.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                 },
             }


             model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                               models=models,param=params)
             
             best_model_score = max(sorted(model_report.values()))

             best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
             ]
             best_model = models[best_model_name]
             save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj= best_model
             )
             predicted = best_model.predict(x_test)
             r2_scoree = r2_score(y_test,predicted)
             return r2_scoree

         except Exception as e:
             raise CustomException(e,sys)