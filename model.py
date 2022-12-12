import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,reca

df=pd.read_csv("diabetes.csv")
#print(df)
X=df[['Pregnancies','Glucose',"BloodPressure","SkinThickness","Insulin","Age","BMI","DiabetesPedigreeFunction"]]
Y=df["Outcome"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y ,test_size=0.2, random_state=0)
scaler=LogisticRegression(random_state=0)
scaler.fit(X_train,Y_train)
y_pred=scaler.predict(X_test)

entree_data=(1,123,23,34,123,12,0.98,65)
def diabete_prediction(entree_data):
    tableau_numpy = np.array(entree_data)
    input_data_reshape = tableau_numpy.reshape(1, -1)
    prediction = scaler.predict(input_data_reshape)

    if (prediction[0] == 1):
        return " La personne est  diabetique"
    else:
        return "La personne n'est pas  diabetique"

diabete_prediction(entree_data)
pickle.dump(scaler, open('scaling.pkl', 'wb'))