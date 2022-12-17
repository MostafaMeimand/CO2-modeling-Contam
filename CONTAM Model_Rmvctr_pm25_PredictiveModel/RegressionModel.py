#%% importing requirements
#%%% Reading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import gurobipy as gp
from gurobipy import GRB
import random
import pwlf
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import math
warnings.filterwarnings("ignore")

#%%
SS_Data = pd.read_excel("30DaysPredictiveModel.xlsx")

X_1 = SS_Data[['Airflow (L/s)', 'Prvs Co2(kg)']]
Y_1 = SS_Data['Next Co2 level']


X_train,X_test,y_train,y_test=train_test_split(X_1,Y_1,test_size=0.3,random_state=3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predicted = lr.predict(X_test)

MSE = np.square(np.subtract(y_test, y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error of temperature model:\n")
print(RMSE)

print(lr.coef_)
print(lr.intercept_)