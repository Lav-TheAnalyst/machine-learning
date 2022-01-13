
import pandas as pd
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('Chemical Process Data/ChemicalProcess.csv')

#mean
imputer=SimpleImputer(strategy='mean')
imp_data=imputer.fit_transform(df)
df1=pd.DataFrame(imp_data,columns=df.columns)

X=df1.drop(['Yield'],axis=1)
y=df1['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))

#median
imputer=SimpleImputer(strategy='mean')
imp_data=imputer.fit_transform(df)
df1=pd.DataFrame(imp_data,columns=df.columns)

X=df1.drop(['Yield'],axis=1)
y=df1['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))


#KNN

imputer=KNNImputer(n_neighbors=1)
imp_data=imputer.fit_transform(df)
df1=pd.DataFrame(imp_data,columns=df.columns)

X=df1.drop(['Yield'],axis=1)
y=df1['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))

imputer=KNNImputer(n_neighbors=2)
imp_data=imputer.fit_transform(df)
df1=pd.DataFrame(imp_data,columns=df.columns)

X=df1.drop(['Yield'],axis=1)
y=df1['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))

imputer=KNNImputer(n_neighbors=3)
imp_data=imputer.fit_transform(df)
df1=pd.DataFrame(imp_data,columns=df.columns)

X=df1.drop(['Yield'],axis=1)
y=df1['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))






from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
imputer= IterativeImputer(random_state=2021,
                         estimator= DecisionTreeRegressor(random_state=2021))
imp_data=imputer.fit_transform(df)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=2021)


rfr=RandomForestRegressor(random_state=2021)
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print(r2_score(y_test, y_pred))



from sklearn.linear_model import LinearRegression
imputer=IterativeImputer(random_state=2021, 
                         estimator=LinearRegression()) 
imp_data=imputer.fit_transform(df) 
df1=pd.DataFrame(imp_data,columns=df.columns) 
 
X=df1.drop(['Yield'],axis=1) 
y=df1['Yield'] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, 
                                               random_state=2021) 
 
 
rfr=RandomForestRegressor(random_state=2021) 
rfr.fit(X_train,y_train) 
y_pred=rfr.predict(X_test) 
print(r2_score(y_test, y_pred))