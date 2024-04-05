import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


heart_data = pd.read_csv("E:/Kenil/Collage/Internship/Project/Backend/Datasets/heart_disease_data.csv")

X = heart_data.drop(columns = 'target', axis = 1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression(solver="liblinear")

model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# print(test_data_accuracy)
# 80.32

pickle.dump(model, open("model_h.pkl", "wb"))

def main(input_data):

    # input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)

    if (prediction[0]== 0):
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'