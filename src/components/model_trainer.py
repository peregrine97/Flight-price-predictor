import sys,os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import customexception
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Split training and test dataset")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],       # all rows only last column
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBoost Regressor":XGBRegressor(),
                "Catboost Regressor":CatBoostRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            score=r2_score(y_test,predicted)

            logging.info("Predicted successfully")
            
            return(
                score,best_model_name
            )
        
        except Exception as e:
            raise customexception(e,sys)


    