import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.metrics import r2_score 

df = pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Concrete Strength/Concrete_Data.csv") 
dum_df = pd.get_dummies(df, drop_first=True) 
 
X = dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1] 

X = X.values 
y = y.values 
 
scaler_x=MinMaxScaler()
X_trn_scl=scaler_x.fit_transform(X)



 
 
scaler_y = MinMaxScaler() 
y_trn_scl = scaler_y.fit_transform(y.reshape(-1,1)) 

model=MLPRegressor(random_state=2021)
kfold=KFold(n_splits=5,random_state=2021,shuffle=True)
params={'hidden_layer_sizes':[(3,2),(4,3,2),(4,2)],
        'solver':['lbfgs','adam','sgd'],
                'learning_rate':['constant','optimal','invscaling','adaptive']}
cv=GridSearchCV(estimator=model,param_grid=params,cv=kfold,verbose=3,scoring='r2')

cv.fit(X_trn_scl,y_trn_scl)
print(cv.best_params_)
print(cv.best_score_)

best_model=cv.best_estimator_
print(best_model.coefs_)
print(best_model.intercepts_)