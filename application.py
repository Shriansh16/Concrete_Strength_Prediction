from flask import Flask,request,render_template,jsonify
import os
import sys
from pathlib import Path 
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            CementComponent=float(request.form.get('CementComponent')),
            BlastFurnaceSlag = float(request.form.get('BlastFurnaceSlag')),
            FlyAsh = float(request.form.get('FlyAsh')),
            Water= float(request.form.get('Water')),
            Superplasticizer= float(request.form.get('Superplasticizer')),
            CoarseAggregate = float(request.form.get('CoarseAggregate')),
            FineAggregate = float(request.form.get('FineAggregate')),
            Age= int(request.form.get('Age'))
            
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictionPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)