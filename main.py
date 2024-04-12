import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from flask import Flask, request, jsonify
from keras.models import save_model, load_model 
import pickle



data = {
    'Self Speed': [50],
    'Right Speed': [55],
    'Left Speed': [45],
    'Left Front Distance (m)': [2.0],
    'Right Front Distance (m)': [2.5],
    'Left Back Distance (m)': [1.8],
    'Right Back Distance (m)': [2.0]
}

def saveModel():
    data = pd.read_csv('./content/simulations.txt')  # Make sure to adjust the path

    # Encode categorical data
    label_encoder_fault = LabelEncoder()
    data['At Fault'] = label_encoder_fault.fit_transform(data['At Fault'])

    label_encoder_prevention = LabelEncoder()
    data['Prevention Measure'] = label_encoder_prevention.fit_transform(data['Prevention Measure'])

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Self Speed', 'Right Speed', 'Left Speed', 'Left Front Distance (m)', 'Right Front Distance (m)', 'Left Back Distance (m)', 'Right Back Distance (m)']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Define features and targets
    X = data[numerical_features]
    y_fault = data['At Fault']
    y_prevention = pd.get_dummies(data['Prevention Measure'])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_fault_train, y_fault_test, y_prevention_train, y_prevention_test = train_test_split(
        X, y_fault, y_prevention, test_size=0.2, random_state=42
    )

    # Input layer
    input_layer = Input(shape=(X_train.shape[1],))

    # Shared layers
    shared = Dense(64, activation='relu')(input_layer)
    shared = Dense(64, activation='relu')(shared)

    # Branch for 'At Fault' prediction
    fault_output = Dense(1, activation='sigmoid', name='fault_output')(shared)

    # Branch for 'Prevention Measure' prediction
    prevention_output = Dense(y_prevention_train.shape[1], activation='softmax', name='prevention_output')(shared)

    # Create model
    model = Model(inputs=input_layer, outputs=[fault_output, prevention_output])

    # Compile the model
    model.compile(optimizer='adam',
                loss={'fault_output': 'binary_crossentropy', 'prevention_output': 'categorical_crossentropy'},
                metrics={'fault_output': 'accuracy', 'prevention_output': 'accuracy'})

    history = model.fit(X_train, {'fault_output': y_fault_train, 'prevention_output': y_prevention_train},
                        validation_split=0.1, epochs=10, batch_size=32)
    model.save("model.h5")
     # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        # Save the label encoder for 'Prevention Measure'
    with open('label_encoder_prevention.pkl', 'wb') as f:
        pickle.dump(label_encoder_prevention, f)

def load_and_predict(input_data_param):
    # Load the model
    model = load_model("model.h5")

    # Assume 'scaler' was fit on the training data during the model training process
    # You need to load or recreate the scaler used during training
    # Load scaler from file if saved during training
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

       # Load the label encoder for 'Prevention Measure'
    with open('label_encoder_prevention.pkl', 'rb') as f:
        label_encoder_prevention = pickle.load(f)

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame(input_data_param)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict with the model
    predictions = model.predict(input_data_scaled)

    fault_prediction = predictions[0]  # This will be your fault status output
    prevention_prediction = predictions[1]  # This will be your prevention measures output

    # If 'At Fault' was binary encoded during training
    fault_status = 'Current' if fault_prediction >= 0.5 else 'Other'

    # If 'Prevention Measure' was one-hot encoded during training
    prevention_measures = label_encoder_prevention.inverse_transform([prevention_prediction.argmax()])[0]

    return {
        "fault": fault_status,
        "prevention": prevention_measures
    }
        

# print(load_and_predict(data))
# saveModel()
app = Flask(__name__)

@app.route('/runmodel', methods=['POST'])
# @app.route('/runmodel')
def hello():
    content = request.json
    data = {
        "Self Speed": [float(content["Self Speed"])],
        "Right Speed": [float(content["Right Speed"])],
        "Left Speed": [float(content["Left Speed"])],
        "Left Front Distance (m)": [float(content["Left Front Distance"])],
        "Right Front Distance (m)": [float(content["Right Front Distance"])],
        "Left Back Distance (m)": [float(content["Left Back Distance"])],
        "Right Back Distance (m)": [float(content["Right Back Distance"])]
    }
    result = load_and_predict(data)
    return jsonify(result)

@app.route('/hello', methods=['POST'])
def test():
    try:
        # return "string"
        return jsonify({"message": "Success"}), 200
    except Exception as e:
        # If an error occurs, return an error response with status code 500
        return jsonify({"error": str(e)}), 500


# # runmodel(data)

if __name__ == '__main__':
    app.run(host='172.20.10.4', debug=True, port= 8080)
