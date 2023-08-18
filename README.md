# Developing-a-Neural-Network-Regression-Model
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## NEURAL NETWORK MODEL:
![6 8](https://github.com/swethamohanraj/Developing-a-Neural-Network-Regression-Model/assets/94228215/4348f2bc-355d-4ee5-b5fe-6ff5abbfaab8)


## DESIGN STEPS:
1.Loading the dataset

2.Split the dataset into training and testing

3.Create MinMaxScalar objects ,fit the model and transform the data.

4.Build the Neural Network Model and compile the model.

5.Train the model with the training data.

6.Plot the performance plot

7.Evaluate the model with the testing data.

## PROGRAM:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DLExp1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

ai=Sequential([
    Dense(6,activation='relu'),
    Dense(8,activation='relu'),
    Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)

## Plot the loss
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```
## DATASET INFORMATION:
![image](https://github.com/swethamohanraj/Developing-a-Neural-Network-Regression-Model/assets/94228215/33961e37-baba-4efc-9491-651e61f5738b)


## OUTPUT:
# Training Loss Vs Iteration Plot:
![image](https://github.com/swethamohanraj/Developing-a-Neural-Network-Regression-Model/assets/94228215/1b08575c-09ee-4e96-bce5-d1e47e2c446b)


## Test Data Root Mean Squared Error:
![image](https://github.com/swethamohanraj/Developing-a-Neural-Network-Regression-Model/assets/94228215/38dd4b71-064f-4c42-8f2d-dde57836ca90)


## New Sample Data Prediction:
![image](https://github.com/swethamohanraj/Developing-a-Neural-Network-Regression-Model/assets/94228215/9c7c18e7-8669-4444-9d0c-4a3de815023c)


## RESULT:
Thus a neural network regression model for the given dataset is written and executed successfully.
