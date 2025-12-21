import sys
import pandas as pd
from src.exception import customexception
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        model_path='artifacts/model.pkl'
        preprocessor_path='artifacts/preprocessor.pkl'
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        return preds
    
class CustomData:
    def __init__(self,
        Airline:str,
        Source:str,
        Destination:str,
        Duration:int,
        Total_Stops:int,
        Dep_Period:str,
        Month:int,
                 ):
        self.Airline=Airline
        self.Source=Source
        self.Destination=Destination
        self.Duration=Duration
        self.Total_Stops=Total_Stops
        self.Dep_Period=Dep_Period
        self.Month=Month

    def get_dataframe(self):
        try:
            custom_input={
                "Airline":[self.Airline],
                "Source":[self.Source],
                "Destination":[self.Destination],
                "Duration":[self.Duration],
                "Total_Stops":[self.Total_Stops],
                "Dep_Period":[self.Dep_Period],
                "Month":[self.Month],
            }
            return pd.DataFrame(custom_input)
        
        except Exception as e:
            raise customexception(e,sys)

        
