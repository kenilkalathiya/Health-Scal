import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

parkinsons_data = pd.read_csv('E:/Kenil/Collage/Internship/Project/Backend/Datasets/parkinsons.csv')

X = parkinsons_data.drop(columns=['status'], axis=1)
Y = parkinsons_data['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# print(test_data_accuracy)
# 87.17

pickle.dump(model, open("model_p.pkl", "wb"))



def main(input_data):
  

#     input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,
# 0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,
# 0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)


    if (prediction[0] == 0):
        print("The Person does not have Parkinsons Disease")

    else:
        print("The Person has Parkinsons")

# input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

#     # changing input data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)

#     # reshape the numpy array
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#     # standardize the data
# std_data = scaler.transform(input_data_reshaped)

# prediction = model.predict(std_data)


# if (prediction[0] == 0):
#         print("The Person does not have Parkinsons Disease")

# else:
#         print("The Person has Parkinsons")