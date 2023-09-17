import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src')
from logger import *
from utils import *
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    training_data_path=os.path.join("artifacts","model_trainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self,training_arr,test_arr):
        try:
            X_train=training_arr[:,:-1]

            y_train=training_arr[:,-1]
            X_test=test_arr[:,:-1]
            y_test=test_arr[:,-1]
            '''
            models={
                "LINEAR_REGRESSION ":LinearRegression(),
                "LASSO ":Lasso(),"Ridge ":Ridge(),
                #"RANDOM_FOREST_REGRESSION ":RandomForestRegressor(),
                "Gradient_boosting_algorithm ":GradientBoostingRegressor(),
                "KNN ":KNeighborsRegressor()
            }
            report=evaluate_model(models,X_train,y_train,X_test,y_test)
            logging.info(f'model report {report}')
            print(report)
            best_model_score=max(list(report.values()))
            best_model_name=list(report.keys())[list(report.values()).index(best_model_score)]
            logging.info(f'BEST MODEL IS {best_model_name} WITH ACCURACY {best_model_score}')
            print('BEST MODEL FOUND ', best_model_name, ' WITH ACCURACY ',best_model_score)
            best_model=models[best_model_name]
            logging.info(f'Shape here {X_train.shape}')
            best_model.fit(X_train,y_train)'''
            rfc=RandomForestRegressor()
            rfc.fit(X_train,y_train)
            y_pred=rfc.predict(X_test)
            print(r2_score(y_test,y_pred))
            ij=rfc.predict([[540.0,0.0,0.0,162.0,2.5,1040.0,676.0,28]])
            lk=rfc.predict([[260.9,100.5,78.3,200.6,8.6,864.5,761.5,28]])
            print(ij)
            print(lk)
            save_object(self.model_trainer_config.training_data_path,rfc)
            #y_pred11=best_model.predict(X_test)
            #logging.info(f'accuaracy {r2_score(y_test,y_pred11)}')
            #save_object(self.model_trainer_config.training_data_path,best_model)

        except Exception as e:
            logging.info("ERROR OCCURED IN MODEL TRAINING")
            print(e)

