import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model/employee-attrition-predict.pk1", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = "Yes" if prediction[0] == 1 else "No"
    return render_template('index.html', prediction_response='Is this employee likely to leave the company: {}'.format(output))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')