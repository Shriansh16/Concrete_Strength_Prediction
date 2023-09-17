import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src\components')
from data_ingestion import *
from data_transformation import *
from model_trainer import *

if __name__=='__main__':

    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    obj1=DataTranformation()
    train_arr,test_arr,_=obj1.initiate_data_transformation(train_path,test_path)
    obj2=ModelTrainer()
    obj2.initiate_model_training(train_arr,test_arr)

