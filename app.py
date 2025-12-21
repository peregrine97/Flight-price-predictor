from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictionPipeline,CustomData

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Airline=request.form.get('Airline'),
            Source=request.form.get('Source'),
            Destination=request.form.get('Destination'),
            Duration=request.form.get('Duration'),
            Total_Stops=request.form.get('Total_Stops'),
            Dep_Period=request.form.get('Dep_Period'),
            Month=request.form.get('Month'),
            
        )
        pred_df=data.get_dataframe()

        predict_pipeline=PredictionPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)