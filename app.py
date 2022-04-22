import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Criar flask app
app = Flask(__name__)

# Carregar o modelo .pkl
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def Home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(i) for i in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    return render_template("index.html", prediction_text = f'A espécie da flor é {labels[prediction[0]]}')


if __name__ == "__main__":
    app.run(debug=True, port=2000)