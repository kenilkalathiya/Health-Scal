# from copyreg import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle


# def trainModel():
diabetes_dataset = pd.read_csv('E:/Kenil/Collage/Internship/Project/Backend/Datasets/diabetes.csv')
diabetes_dataset.head()

diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

pickle.dump(model, open("model_d.pkl", "wb"))

# trainModel()

def detectDiabetesFunc(input_data):
    print(input_data)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)
    return prediction[0]
  


# input_data = (5,166,72,19,175,25.8,0.587,51)
# input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
# print(main(input_data))

# # changing the input_data to numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# # standardize the input data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data)

# prediction = model.predict(std_data)
# print(prediction)

# if (prediction[0] == 0):
#   print('The person is not diabetic')
# else:
#   print('The person is diabetic')