from flask import Flask, render_template, json, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pickle

import os

from sklearn.metrics import f1_score

app = Flask(__name__)
model = pickle.load(open('./data/model.pkl', 'rb'))


@app.route('/')
def index():
	return render_template('index.html')


#first 4 rows rows
@app.route('/data/test', methods=['GET'])

def fetch():
	with open('./data/test.csv', 'r') as file:
		data = file.readlines()[1:5]
		for info in data:
			print(info)
			return jsonify(data)

#prediction API
@app.route('/predict',methods=['POST'])
def predict():
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values()))])

	output = prediction[0]
	return jsonify(output)
   			

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))