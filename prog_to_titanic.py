import pandas as pd
import numpy as np

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data_train_path='/home/devel/python_prog/Titanic/train.csv'
data_test_path='/home/devel/python_prog/Titanic/test.csv'
dataFrame_train=pd.read_csv(data_train_path)
dataFrame_test=pd.read_csv(data_test_path)

Age_Fare=dataFrame_train.Age*dataFrame_train.Fare
Age_Fare.name='AgeFare'
dataFrame_train=dataFrame_train.join(Age_Fare)
Y=dataFrame_train.Survived
X_Features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','AgeFare']
X=dataFrame_train[X_Features]


imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
For_Pre=X.Age.values
For_Pre=For_Pre.reshape(-1,1)
imp.fit(For_Pre)
For_Pre=imp.transform(For_Pre)
X.Age=For_Pre
print(X)

For_Pre=X.AgeFare.values
For_Pre=For_Pre.reshape(-1,1)
imp.fit(For_Pre)
For_Pre=imp.transform(For_Pre)
X.AgeFare=For_Pre
print(X)

#X['Title']= dataFrame_train[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

#X[ 'Title' ] = X.Title.map( Title_Dictionary )

#X.Cabin=X.Cabin.fillna('U')
#X.Cabin=X.Cabin.map(lambda x:x[0])


X=pd.get_dummies(X)

X[ 'FamilySize' ] = X[ 'Parch' ] + X[ 'SibSp' ] + 1

X[ 'Family_Single' ] = X[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
X[ 'Family_Small' ]  = X[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
X[ 'Family_Large' ]  = X[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
print(X)

X.pop('SibSp')
X.pop('Parch')

X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,random_state=0)



#my_model_reg=DecisionTreeRegressor()
#my_model_reg.fit(X_train,Y_train)
#pred=my_model_reg.predict(X_valid)
#print('MAE DTR: ',mean_absolute_error(pred,Y_valid))

class_model=DecisionTreeClassifier(max_leaf_nodes=54)
class_model.fit(X_train,Y_train)
pred_class=class_model.predict(X_valid)
print('MAE DTC: ', mean_absolute_error(pred_class,Y_valid))



SGD_model=SGDClassifier()
SGD_model.fit(X_train,Y_train)
pred_SGD=SGD_model.predict(X_valid)
print('MAE SGD: ', mean_absolute_error(pred_SGD,Y_valid))


RF_model=RandomForestClassifier(max_depth=17,random_state=0)
RF_model.fit(X_train,Y_train)
pred_RF=RF_model.predict(X_valid)
print('MAE RF: ', mean_absolute_error(pred_RF,Y_valid))


KN_model=KNeighborsClassifier(n_neighbors=55,weights='distance',algorithm='auto')
KN_model.fit(X_train,Y_train)
pred_KN=KN_model.predict(X_valid)
print('MAE KN: ',mean_absolute_error(pred_KN,Y_valid))




RN_model=RadiusNeighborsClassifier(radius=3.32,weights='distance',algorithm='ball_tree',outlier_label=1)
RN_model.fit(X_train,Y_train)
pred_RN=RN_model.predict(X_valid)
print('MAE RAD_N: ',mean_absolute_error(pred_RN, Y_valid))

GB_model=GradientBoostingClassifier(learning_rate=0.0730,n_estimators=250)#(learning_rate=0.028,n_estimators=375)
Fitted_GB=GB_model.fit(X_train,Y_train)
pred_GB=GB_model.predict(X_valid)
print('MAE GB: ',mean_absolute_error(pred_GB, Y_valid))

score=(cross_val_score(GB_model,X,Y)).mean()
print('CROSS-VALIDATION_GB= ',score)

SVM_model=SVC()
SVM_model.fit(X_train,Y_train)
pred_SVM=SVM_model.predict(X_valid)
print('MAE SVM:',mean_absolute_error(pred_SVM,Y_valid))


DTC_accuracy=accuracy_score(pred_class, Y_valid)
SGD_accuracy=accuracy_score(pred_SGD, Y_valid)
RF_accuracy=accuracy_score(pred_RF, Y_valid)
KN_accuracy=accuracy_score(pred_KN, Y_valid)
RAD_N_accuracy=accuracy_score(pred_RN, Y_valid)
GB_accuracy=accuracy_score(pred_GB, Y_valid)
print('Accuracy:\nDTC=%.6f;\nSGD=%.6f;\nRF=%.6f;\nKN=%.6f;\nRN=%.6f;\nGB=%.6f;\n'%(DTC_accuracy,SGD_accuracy,RF_accuracy,KN_accuracy,RAD_N_accuracy,GB_accuracy))




input("Later...!!!")
#scp=pd.scatter_matrix(X,c=Y,figsize=(10,10),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)


#FOR DEBUGGING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#weights_KN=('learning_rate','distance')
#algorithm_KN=('ball_tree','kd_tree','brute','auto')
min_KN=1.0
#for i in range(1,1):
#    for j in weights_KN:
i=0.001
while i<=1.0:
        for k in range(100,500,19):
            _model=GradientBoostingClassifier(learning_rate=i,n_estimators=k)
            _model.fit(X_train,Y_train)
            pred_=_model.predict(X_valid)
#            print(i,'#',k)
            if min_KN>mean_absolute_error(pred_,Y_valid):
                min_KN=mean_absolute_error(pred_,Y_valid)
                print('MAE GB:%.6f;(lr= %.4f; estim= %d;)'%(min_KN,i,k))
        i+=0.010

#MY_INPUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#activation=('identity', 'logistic') #, 'tanh', 'relu')
#min_KN=1.0
#for i in range(1,1):
#    for j in weights_KN:
#i=0.001
#while i<=1.0:
#        for k in range(100,200,20):
#          for j in activation: 
#            _model=MLPClassifier(hidden_layer_sizes=(k,),learning_rate='constant',learning_rate_init=i,solver='lbfgs',activation=j)
#            _model.fit(X_train,Y_train)
#            pred_=_model.predict(X_valid)
#            print(i,'#',k)
#            if min_KN>mean_absolute_error(pred_,Y_valid):
#                min_KN=mean_absolute_error(pred_,Y_valid)
#                print('MAE GB:%.6f;(lr= %.4f; hdl= %d;act= %s)'%(min_KN,i,k,j))
#        i+=0.02



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



