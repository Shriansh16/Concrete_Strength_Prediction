import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, 'D:\Cement_Strength_Prediction\src')
from logger import *
from utils import *



class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessing.pkl')
            model_path=os.path.join('artifacts','model_trainer.pkl')
            
            preprocessor=open_object(preprocessor_path)
            
            model=open_object(model_path)
            
            scaled_data=preprocessor.fit_transform(features)
            
            prediction=model.predict(scaled_data)
            
            
            return prediction
        except Exception as e:
            logging.info("ERROR OCCURED IN PREDICTION")
            print(e)
class CustomData:
    def __init__(self,CementComponent:float,BlastFurnaceSlag:float,FlyAsh:float,Water:float,Superplasticizer:float,CoarseAggregate:float,FineAggregate:float,Age:int):
        self.CementComponent = CementComponent
        self.BlastFurnaceSlag = BlastFurnaceSlag
        self.FlyAsh = FlyAsh
        self.Water = Water
        self.Superplasticizer = Superplasticizer
        self.CoarseAggregate = CoarseAggregate
        self.FineAggregate = FineAggregate
        self.Age = Age

    def get_data_as_dataframe(self):
        try:
            CustomDataInput={
                'CementComponent':[self.CementComponent],'BlastFurnaceSlag':[self.BlastFurnaceSlag],'FlyAsh':[self.FlyAsh],
                'Water':[self.Water],'Superplasticizer':[self.Superplasticizer],'CoarseAggregate':[self.CoarseAggregate],
                'FineAggregate':[self.FineAggregate],'Age':[self.Age]
            }
            df=pd.DataFrame(CustomDataInput)
            return df
            

        except Exception as e:
            logging.info("ERROR OCCURED IN GATHERING THE DATA")
            print(e)



