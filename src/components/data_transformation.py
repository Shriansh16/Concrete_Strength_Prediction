import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src')
from logger import *
from utils import *
from dataclasses import dataclass

@dataclass
class DataTransforamtionConfig:
    preprocessing_data_path=os.path.join("artifacts","preprocessing.pkl")

class DataTranformation:
    def __init__(self):
        self.data_tramsforamtion_config=DataTransforamtionConfig()

    def get_data_transforamtion(self):
        try:
            scaler=StandardScaler()
            return scaler
        except Exception as e:
            logging.info("ERROR OCCURED IN DEFINING THE PREPROCESSOR")
            print(e)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info(f'{train_data.head()}')
            logging.info(f'{test_data.head()}')
            logging.info("train data and test data taken")
            logging.info(f'1 {train_data.shape}')
            X_train=train_data.iloc[:,:-1]
            logging.info(f'shape {X_train.shape}')
            y_train=train_data.iloc[:,-1]
            X_test=test_data.iloc[:,:-1]
            y_test=test_data.iloc[:,-1]
            scaler1=self.get_data_transforamtion()
            X_train_trans=scaler1.fit_transform(X_train)
            X_test_trans=scaler1.transform(X_test)
            training_arr=np.c_[X_train_trans,np.array(y_train)]
            test_arr=np.c_[X_test_trans,np.array(y_test)]
            save_object(self.data_tramsforamtion_config.preprocessing_data_path,scaler1)
            return (
                training_arr,test_arr,self.data_tramsforamtion_config.preprocessing_data_path
            )
        except Exception as e:
            logging.info("ERROR OCCURED DURING DATA TRANSFORMATION")
            print(e)

