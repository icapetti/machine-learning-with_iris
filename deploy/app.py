import pickle, flask, os
import pandas as pd

# Creates the Flask application object
app = flask.Flask(__name__)
port = int(os.getenv('PORT', 9099))

# Imports the model file
model = pickle.load(open('iris/deploy/svm_model.pkl', 'rb'))

# Adds the route that will allow send a JSON body features and will return a prediction
@app.route('/predict', methods=['POST'])

def predict():
    features = flask.request.get_json(force=True)['features']
    print('features received: ',features)
    features_df = pd.DataFrame(columns=['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width'])

    new_row = {'sepal_lenght':features[0]
            ,'sepal_width':features[1]
            ,'petal_lenght':features[2]
            ,'petal_width':features[3]}

    features_df = features_df.append(new_row, ignore_index=True)

    prediction = model.predict(features_df)
    print('prediction sent: ', prediction)
    response = {'prediction':int(prediction)}
    print(response)

    return flask.jsonify(response)

if __name__ == '__main__':
    # A method that runs the application server.
    app.run(host='0.0.0.0', port=port)