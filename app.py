import pickle
from flask import Flask , request , app , jsonify , url_for , render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

# load the model
regmodel = pickle.load(open('regmodel.pkl' , 'rb'))
scalar = pickle.load(open('scaling.pkl' , 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api' , methods=['GET','POST'])
def predict_api():
     
     if request.method=='POST':
        CRIM=float(request.form.get('CRIM'))
        ZN = float(request.form.get('ZN'))
        INDUS = float(request.form.get('INDUS'))
        CHAS = float(request.form.get('CHAS'))
        NOX = float(request.form.get('NOX'))
        RM = float(request.form.get('RM'))
        AGE = float(request.form.get('AGE'))
        DIS = float(request.form.get('DIS'))
        RAD = float(request.form.get('RAD'))
        TAX = float(request.form.get('TAX'))
        PTRATIO = float(request.form.get('PTRATIO'))
        B = float(request.form.get('B'))
        LSTAT = float(request.form.get('LSTAT'))

        new_data = scalar.transform([[CRIM , ZN , INDUS , CHAS , NOX , RM , AGE , DIS , RAD , TAX , PTRATIO , B , LSTAT]])

        output = regmodel.predict(new_data)

        return render_template('predict.html',output=output[0])
     
     else:
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
