from flask import Flask,request,app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# scaler=pickle.load(open("Model/scaler.pkl","rb"))
# clf = pickle.load(open("Model/lregression.pkl", "rb"))

def load_model():
    # Load the scaler
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))

    # Load the logistic regression model
    clf = pickle.load(open("Model/lregression.pkl", "rb"))

    # Check if the model has been fitted
    if not hasattr(clf, "classes_"):
        # Create a more meaningful dummy input with samples from both classes
        X_dummy = np.vstack([np.zeros((1, len(scaler.mean_))), np.ones((1, len(scaler.mean_)))])
        y_dummy = np.array([0, 1])

        # Fit the model with the dummy input
        clf.fit(X_dummy, y_dummy)

        # Print information about the model
        print("Classes:", clf.classes_)
        print("Coef:", clf.coef_)

    return scaler, clf





scaler, clf = load_model()  # Load the model and scaler


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/Predictdata',methods=['GET','POST'])
def predict_datapoint():
   result=""
   if request.method== "POST":
     Pregrenancies=int(request.form.get("Pregrenancies"))
     Glucose=float(request.form.get('Glucose'))
     BloodPressure=float(request.form.get('BloodPressure'))
     SkinThickness=float(request.form.get('SkinThickness'))
     Insulin=float(request.form.get('Insulin'))
     BMI=float(request.form.get('BMI'))
     DiabeterPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
     Age=float(request.form.get('Age'))

     new_data=scaler.transform([[Pregrenancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabeterPedigreeFunction,Age]])
     predict=clf.predict(new_data)


     if predict[0]==1 :
        result='Diabetic'
     else: 
        result='Non-Diabetic'

     return render_template('single_prediction.html',result=result)
   else:
    return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")