import h2o
h2o.init()
from h2o.estimators.glm import H2OGeneralizedLinearEstimator 
df=h2o.import_file("C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Bankruptcy/Bankruptcy.csv")
print(df.col_names)

y= 'D' 
x = df.col_names[2:] 
 
df['D'] = df['D'].asfactor() 
df['D'].levels() 
train,test = df.split_frame(ratios=[.7],seed=2021) 
print(train.shape) 
print(test.shape) 
 
glm_model = H2OGeneralizedLinearEstimator(family='binomial') 
glm_model.train(x=X,y=y,training_frame=train,validation_frame=test) 
 
 
y_pred = glm_model.predict(test_data=test) 
y_pred_df = y_pred.as_data_frame() 
 
 
print(glm_model.auc()) 
print(glm_model.confusion_matrix())