import numpy as np
import pandas as pd
import dill 
import os,sys
from src.exception import customexception
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise customexception(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(x_train,y_train)

            y_pred=model.predict(x_test)
            acc=r2_score(y_test,y_pred)

            report[list(models.keys())[i]]=acc
        
        return report
    except Exception as e:
        raise customexception(e,sys)