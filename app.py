import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

test=pd.read_csv("test_data.csv",error_bad_lines=False)
x_test=test.drop('prognosis',axis=1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    col=x_test.columns
    inputt=[]
    for x in request.form.values():
        inputt.append(x)

    b=[0]*132
    for x in range(0,132):
        for y in inputt:
            if(col[x]==y):
                b[x]=1
    b=np.array(b)
    b=b.reshape(1,132)
    prediction = model.predict(b)

    return render_template('index.html', prediction_text='The disease is {}')


if __name__ == "__main__":
    app.run(debug=True)