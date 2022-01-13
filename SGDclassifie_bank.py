import pandas as pd 
from sklearn.model_selection import GridSearchCV,KFold 
from sklearn.linear_model import SGDClassifier 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold 

df = pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/bank/bank.csv",sep=';') 
dum_df = pd.get_dummies(df, drop_first=True) 
X = dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1] 
 
 
X = X.values 
y = y.values 
 
 
 
 
scaler_X = MinMaxScaler() 
X_trn_scl = scaler_X.fit_transform(X) 
 
 
scaler_y = MinMaxScaler() 
y_trn_scl = scaler_y.fit_transform(y.reshape(-1,1)) 
 
model = SGDClassifier(random_state=2021) 
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)  
params = {'penalty':['l1','l2','elasticnet'], 
          'eta0':[0.1,0.3,0.4],
          'learning_rate':['constant','optimal','invscaling','adaptive']} 
cv = GridSearchCV(estimator=model, param_grid=params, 
                  cv=kfold,verbose=3,scoring='roc_auc') 
cv.fit(X_trn_scl,y) 
 
 
print(cv.best_params_) 
print(cv.best_score_)