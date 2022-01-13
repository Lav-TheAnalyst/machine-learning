import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score 
 
 
df = pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Concrete Strength/Concrete_Data.csv") 
dum_df = pd.get_dummies(df, drop_first=True) 
 
X = dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1] 

X = X.values 
y = y.values 
 
 
X_train,X_test,y_train,y_test =train_test_split(X,y, 
                                                test_size=0.3, 
                                                random_state=2021) 
 
 
 
scaler_X = MinMaxScaler() 
X_trn_scl = scaler_X.fit_transform(X_train) 
 
 
scaler_y = MinMaxScaler() 
y_trn_scl = scaler_y.fit_transform(y_train.reshape(-1,1)) 
for i in range(1,7):
    knn = KNeighborsRegressor(n_neighbors=i) 
    knn.fit(X_trn_scl,y_trn_scl) 
    X_tst_scl = scaler_X.transform(X_test) 
    y_pred_scl = knn.predict(X_tst_scl) 
    y_pred = scaler_y.inverse_transform(y_pred_scl)
    print(i)
    print(r2_score(y_test,y_pred))