from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open('Models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    def get_float(val, default=0.0):
        return float(val) if val != "" else default

    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = get_float(request.form['age'])
    fare = get_float(request.form['fare'])
    embarked = int(request.form['embarked'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])

    family = sibsp + parch + 1
    is_alone = 0 if family > 1 else 1

    final = np.array([[pclass, sex, age, fare, embarked, family, is_alone]])
    pred = model.predict(final)

    result = "Survived" if pred[0] == 1 else "Not Survived"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
