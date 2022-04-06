from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST","GET"])
def predict():
    # Check if request has a JSON content and if it is a POST method
    if request.method == "POST" and request.is_json:
        # Get the JSON as dictionnary
        req = request.get_json()
        
        # Check mandatory key
        if "input" in req.keys() :
            # Load model
            regressor = joblib.load("model.joblib")
            #prepare the data
            inp=np.array(req["input"])
            # Predict
            prediction = regressor.predict(inp)
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            prediction = prediction.tolist()
            return jsonify({"The note will be": prediction}), 200
        else:
            return jsonify({"msg": "No Input key found in your request"})

    elif request.method == "GET":
        return jsonify({"msg": "Only the POST Methods are accepted"}), 200
    else:
        return jsonify({"msg": "Error: not a JSON in your request"})

@app.route("/")
def index():
    return render_template("index.html")

 
if __name__ == "__main__":
    app.run(debug=True)