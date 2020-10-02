#Regression models with randomly creeated data.
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import random


def create_y_data(variance):
    y_data = []
    for x in range(1000):
        y_data.append(x + (random.randint(0,variance)))
    return y_data 

def multi_regression(y_data, variance):
    x_data = []
    for x in range(1000):
        x_data.append(x)
    x_data = np.array(x_data).transpose()
    y_data = np.array(y_data).transpose()
    y_train = y_data[:-200]
    y_test =  y_data[-200:]
    x_train = x_data[:-200]
    x_test = x_data[-200:]
    
    
    
    linreg = linear_model.LinearRegression()
    ridgereg = linear_model.Ridge()
    lassoreg = linear_model.Lasso()
    
    linreg.fit(x_train.reshape(-1,1), y_train)
    ridgereg.fit(x_train.reshape(-1,1), y_train)
    lassoreg.fit(x_train.reshape(-1,1), y_train)
    
    y_prediction_lin = linreg.predict(x_test.reshape(-1,1))
    y_prediction_ridge = ridgereg.predict(x_test.reshape(-1,1))
    y_prediction_lasso = lassoreg.predict(x_test.reshape(-1,1))
    
    MSElin = metrics.mean_squared_error(y_test,y_prediction_lin)
    MSEridge = metrics.mean_squared_error(y_test,y_prediction_ridge)
    MSElasso = metrics.mean_squared_error(y_test,y_prediction_lasso)
    
    r2scorelin = metrics.r2_score(y_test,y_prediction_lin)
    r2scoreridge = metrics.r2_score(y_test,y_prediction_ridge)
    r2scorelasso = metrics.r2_score(y_test,y_prediction_lasso)
    
    
    figlin, axlin = plt.subplots(3)
    axlin[0].scatter(x_test, y_test, color = "black")
    axlin[0].plot(x_test, y_prediction_lin, color = "red")
    axlin[0].set_title("Linear Regression, Rnd Y Change = %s" %variance)
    
    axlin[1].scatter(x_test, y_test, color = "black")
    axlin[1].plot(x_test, y_prediction_ridge, color = "blue")
    axlin[1].set_title("Ridge Regression, Rnd Y Change = %s" %variance)
    
    axlin[2].scatter(x_test, y_test, color = "black")
    axlin[2].plot(x_test, y_prediction_lasso, color = "green")
    axlin[2].set_title("Lasso Regression, Rnd Y Change = %s" %variance)
   
    
    plt.show()
    
    print("Mean Squared Error (Linear)  = %.2f" %MSElin)
    print("R2 Score (Linear) = %.2f" %r2scorelin)
    print("\n")
    print("Mean Squared Error (Ridge)  = %.2f" %MSEridge)
    print("R2 Score (Ridge) = %.2f" %r2scoreridge)
    print("\n")
    print("Mean Squared Error (Lasso)  = %.2f" %MSElasso)
    print("R2 Score (Lasso) = %.2f" %r2scorelasso)


multi_regression(create_y_data(1), "1")

multi_regression(create_y_data(10), "10")

multi_regression(create_y_data(50), "50")

multi_regression(create_y_data(100), "100")