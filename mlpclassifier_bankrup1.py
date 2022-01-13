import pandas as pd  
#from sklearn.model_selection import GridSearchCV,StratifiedKFold 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler  
 
  
df = pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/bank/bank.csv",sep=';')
dum_df=pd.get_dummies(df,drop_first=True) 
 
X = dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1] 
 
  
X = X.values  
y = y.values  
  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2021) 
  
scaler_X = MinMaxScaler()  
X_trn_scl = scaler_X.fit_transform(X)  
 
model = MLPClassifier(random_state=2021,hidden_layer_sizes=(2,), 
                      activation="Logistic") 
 
model.fit(X_trn_scl,y_train) 
X_tst_scl =scaler_X.transform(X_test) 
y_pred_scl=model.predict(X_tst_scl)