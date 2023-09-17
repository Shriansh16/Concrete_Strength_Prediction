import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.metrics import r2_score
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src')
from logger import *


def remove_outliers(df,col):
    q1=np.percentile(df[col],25)
    q3=np.percentile(df[col],75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    return df[(df[col]<=upper_bound) & (df[col]>=lower_bound)]

def save_object(path,object):
    try:
        
        with open(path,'wb') as obj:
            pickle.dump(object,obj)
    except Exception as e:
        logging.info("ERROR OCCURED IN SAVING THE MODEL")
        print(e)

def open_object(path):
    try:
        with open(path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info("ERROR OCCURED IN LOADING THE OBJECT")
        print(e)

def evaluate_model(models,X_train,y_train,X_test,y_test):
    try:
        reports={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            accuracy=r2_score(y_test,y_pred)
            reports[list(models.keys())[i]]=accuracy
        return reports
    except Exception as e:
        logging.info("ERROR OCCURED IN SELECTING THE MODEL")
        print(e)
        
            


        

     