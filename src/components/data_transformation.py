import sys,os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import customexception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            categorical_columns=['Airline', 'Source', 'Destination', 'Dep_Period']
            numerical_columns=['Duration', 'Total_Stops', 'Month']

            num_pipeline=Pipeline(
                steps=[
                    # ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("numerical scaling is completed")

            cat_pipeline=Pipeline(
                steps=[
                    # ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
                ]
            )

            logging.info("categorical encoding is completed")

            preprocessor=ColumnTransformer([
                ("numerical",num_pipeline,numerical_columns),
                ("categorical",cat_pipeline,categorical_columns),
            ])

            return preprocessor
        except Exception as e:
            raise customexception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and test datas read successfully")
            
            preprocessor_obj=self.get_data_transformer_obj()

            logging.info("Preprocessor object obtained succesfully")

            target_column='Price'
            categorical_columns=['Airline', 'Source', 'Destination', 'Dep_Period']
            numerical_columns=['Duration', 'Total_Stops', 'Month']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)

            target_feature_train_df=train_df[target_column]
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprocessor object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            # if len(target_feature_train_df) != input_feature_train_arr.shape[0]:
    # Something removed rows - find out why
            logging.info(f"{len(target_feature_train_df)}")
            logging.info(f"{input_feature_train_arr.shape[0]}")


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing project.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise customexception(e,sys)


