import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

data_train_path='/home/devel/python_prog/Titanic/train.csv'
data_test_path='/home/devel/python_prog/Titanic/test.csv'
dataFrame_train=pd.read_csv(data_train_path)
dataFrame_test=pd.read_csv(data_test_path)


Y=dataFrame_train.Survived
X_Features=['Pclass','Sex','Age']
X=dataFrame_train[X_Features]


imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
For_Pre=X.Age.values
For_Pre=For_Pre.reshape(-1,1)
imp.fit(For_Pre)
For_Pre=imp.transform(For_Pre)
X.Age=For_Pre
print(X)


X=pd.get_dummies(X)
print(X)


X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,random_state=0)



#my_model_reg=DecisionTreeRegressor()
#my_model_reg.fit(X_train,Y_train)
#pred=my_model_reg.predict(X_valid)
#print('MAE DTR: ',mean_absolute_error(pred,Y_valid))

  
class_model=DecisionTreeClassifier(max_leaf_nodes=7)
class_model.fit(X_train,Y_train)
pred_class=class_model.predict(X_valid)
print('MAE DTC: ', mean_absolute_error(pred_class,Y_valid))


SGD_model=SGDClassifier()
SGD_model.fit(X_train,Y_train)
pred_SGD=SGD_model.predict(X_valid)
print('MAE SGD: ', mean_absolute_error(pred_SGD,Y_valid))


RF_model=RandomForestClassifier(max_depth=3,random_state=0)
RF_model.fit(X_train,Y_train)
pred_RF=RF_model.predict(X_valid)
print('MAE RF: ', mean_absolute_error(pred_RF,Y_valid))



KN_model=KNeighborsClassifier(n_neighbors=18,weights='distance',algorithm='ball_tree')
KN_model.fit(X_train,Y_train)
pred_KN=KN_model.predict(X_valid)
print('MAE KN: ',mean_absolute_error(pred_KN,Y_valid))


RN_model=RadiusNeighborsClassifier(radius=1.8,weights='distance',algorithm='ball_tree',outlier_label=1)
RN_model.fit(X_train,Y_train)
pred_RN=RN_model.predict(X_valid)
print('MAE RAD_N: ',mean_absolute_error(pred_RN, Y_valid))


GB_model=GradientBoostingClassifier(learning_rate=0.011,n_estimators=100)
Fitted_GB=GB_model.fit(X_train,Y_train)
pred_GB=GB_model.predict(X_valid)
print('MAE GB: ',mean_absolute_error(pred_GB, Y_valid))





DTC_accuracy=accuracy_score(pred_class, Y_valid)
SGD_accuracy=accuracy_score(pred_SGD, Y_valid)
RF_accuracy=accuracy_score(pred_RF, Y_valid)
KN_accuracy=accuracy_score(pred_KN, Y_valid)   
RAD_N_accuracy=accuracy_score(pred_RN, Y_valid)
GB_accuracy=accuracy_score(pred_GB, Y_valid)
print('Accuracy:\nDTC=%.6f;\nSGD=%.6f;\nRF=%.6f;\nKN=%.6f;\nRN=%.6f;\nGB=%.6f;\n'%(DTC_accuracy,SGD_accuracy,RF_accuracy,KN_accuracy,RAD_N_accuracy,GB_accuracy))




input("Later...")

#FOR DEBUGGING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#weights_KN=('learning_rate','distance')
#algorithm_KN=('ball_tree','kd_tree','brute','auto')
min_KN=1.0
#for i in range(1,1):
#    for j in weights_KN:
i=0.18
while i<=1.0:
        for k in range(100,600,5):
            _model=GradientBoostingClassifier(learning_rate=i,n_estimators=k)
            _model.fit(X_train,Y_train)
            pred_=_model.predict(X_valid)
            print(i,'#',k)
            if min_KN>mean_absolute_error(pred_,Y_valid):
                min_KN=mean_absolute_error(pred_,Y_valid)  
                print('MAE GB:%.6f;(lr= %.4f; estim= %d;)'%(min_KN,i,k))
        i+=0.01

#MY_INPUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#my_new=[]
#st_in=input("enter in comma Pclass,Age,Sex\n:  ")
#my_new=st_in.split(sep=',')
#if st_in[2]=='male':
#  my_new.pop(2)
#  my_new.append(0)
#  my_new.append(1)
#else:
#  my_new.pop(2)
#  my_new.append(1)
#  my_new.append(0)
#my_new=pd.Series(my_new)
#print("DTC: %d;\nSGD: %d;\nRF: %d;\n"%(class_model.predict([my_new]),SGD_model.predict([my_new]),RF_model.predict([my_new])))



