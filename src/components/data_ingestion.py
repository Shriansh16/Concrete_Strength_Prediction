import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src')
from logger import *
from utils import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','raw.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.Data_Ingestion_Config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:

            logging.info("DATA INGESTION STARTED")
            data=pd.read_csv(os.path.join('notebooks','Cement_data.csv'))
            logging.info("dataset read as pd")
            for i in data.columns:
                final_data=remove_outliers(data,i)
            logging.info(f'{final_data.shape}')
            final_data.reset_index(drop=True,inplace=True)
            logging.info(f'{final_data.head()}')
            logging.info("OUTLIERS ARE REMOVED")
            logging.info(f'{final_data.shape}')
            final_data.to_csv(self.Data_Ingestion_Config.raw_data_path,index=False)
            train_data,test_data=train_test_split(final_data,test_size=0.20,random_state=42)
            train_data.to_csv(self.Data_Ingestion_Config.train_data_path,index=False)
            test_data.to_csv(self.Data_Ingestion_Config.test_data_path,index=False)
            return(self.Data_Ingestion_Config.train_data_path,
                   self.Data_Ingestion_Config.test_data_path

            )
        except Exception as e:
            logging.info("ERROR OCCURED IN DATA INGESTION")
            print(e)
