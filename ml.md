Linear regression 
y=mx+c


Regression matrics
1) MSE(Mean Squarred Error) 
2) RMSE(Root Mean Squarred Error)
3) MAE(Mea Absolute Error)
4) R^2 Score
Error=Y-Y^
y-->Actual output
y^-->Predicted output

R^2 score --> it is regression matrics it tells how will model captures the pattern

variance->Squared units of target column X or y
Stadard deviation -> actual unit of X or Y


Non Linear regression
it learns pattern from unlabelled data
How to handle non linear data?
1)Polynomial regression
2)Tree based algroithms
  decision Tree
  random forest
  xgboost

Polynomial regression: it is preproccesing technique which help to handle non linear data pattern by adding extra features with the help of degree
#fit_transform()
#transform()
degree=1 -> mx+c
degree=2 -> mx+mx^2+c 
degree=3 -> mx+mx^2+mx^3+c


Scaling: it is a preprocessing technique it scales the values equally for all the columns
when we use scaling?
1)Only for numerical columns not for objective columns
2)when numerical column values are in different range then use scaling technique

Scaling
1)Standardization
   Standard scale
   Robust scale
2)Normalization
  minmax scale


eg
 AGE        SAL 
  28       25000
  26       30000
  22       45000
  10       50000
  18       19000
both the columns are different range so using the scaling technique
For tree based algorithm scaling techniques are not required

Standardization: it is a type of scaling preproccesing technique but their no fix range values. this is used for outliers
  standard scale: when we used this have less outliers by using boxplot(use z score formula)
  Robust scale: it is used mean and median
Normalization: it is a type of scaling preproccesing technique but their fix range values(0-1). this is used for when data is clean
  minmax scale: 



Enoding: it is ml preproccesing technique which converts categerical values into numericals
1)label encoder
2)one hot encoder
3)ordinal encoder
4)target encoder

1)label encoder: it is a enoding preprocessing technique which accepts one column and encodes the values from zero it access only one column
 from sklearn.preprocessing import LabelEncoder
 encodes=LabelEncoder()
 encodes.fit_transform(Xtrain[column])
encodes.fit_transform(Xtrain[column])
encoder.transform(Xtest[colname])
encoder.transform(Xtest[colname])

when to use label encoding?
*when we have only one categerical column 
*for less unique values
*mostly we use this for target column(classification scenarios)

when not to use label encoder?
*when we have two or more categerical columns
*when we have more unique values


2)One Hot encoder: it is a encoding preproccesing technique which accepts 2D and encodes binary values
when to use one hot encoder?
from sklearn.preprocessing import OneHotEncoder
 encodes=OneHotEncoder(handle_unknown='ignore'|'error'(default),sparse_output=True|False default)
 encodes.fit_transform(Xtrain[column])
encodes.fit_transform(Xtrain[column])
encoder.transform(Xtest[colname])
encoder.transform(Xtest[colname])
*when feature columns are having 2 or more columns then we use one hot encoder
*for less unique values

when not we use one hot encoder?
*when we have more unique values

Onehotencoder it handles the unknown values with 'ignore','error'


3)ordinal encoder: it is encoding preproccesing technique which encodes the values based on the order
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder(categories=[[col1 order],[col2 order]])
                                  [['A1','B1','A12','B2']
                                  ['High school','UG','PG','PhD']]
encoder.fit_transform(Xtrain)

4)Target encoder: it is an encoding preproccesing technique which encodes the values based on target column
if target column is continuous valuse then it takes average 
if the target column is categerical values then it takes majority
from Catergory_encoder import TargetEncoder
encoder=TargetEncoder()
encoder.fit_transform(Xtrain)
encoder.transform(Xtest,ytest)


---------------------------------------------------------------------------------------
LOGISTIC REGRESSION:
it is a supervised learning algorithm which comes under regression and use for classification scenario
ex:Spam detection(target: spam/not spam)
  Sentiment Analysis(predicts positive,negative,neutral)

what is classification algroithms?
it is a supervised learning technique which Predicts categerical values

how to select algorithm?
if target column values are continuous then regression algorithm
if target column values are categerical then use classification algorithm

why its calls regression?
its internally use regression formula for Prediction

Sigmoid function 
binary class label(target column values)
p=1/(1+e^-z)
>=0.5-->1
<=0.5-->0

How weights are calculating in LOGISTIC regression
*first randomly initialize the random weight between 0 and 0.5
*and later with the help of gradient descent it finds the best weights
*to find or optimize the weights internally takes multiple iterations

what and all techniques LOGISTIC regression use internally?
*loss function
*cost function
*Sigmoid function
*regularization techniques(l1,l2,elastic net)
*gradient descent
 
regularization: it is ml technique which help us to reduce the gradient with the help of penalties(l1,l2,elastic net)
l1->Lasso: 
*it is a regularization penalty which selects important features by pushing the weights towards to 0 and exactly to 0(with the help of gradient descent it pushes weights towards to 0)
*if the weight is 0 which means that feature is not important
m1=0,m2=5,c=10,x1=5,x2=6
y=m1x1+m2x2+c 
y=0*5+5*6+10
When to use lasso?
*when we have more features 
*for sentence based classification
*when we dont know the important features  
l2->Ridge: 
*it is regularization penalty which keeps all the columns by strings the weights
*it strings the weights but never pushes to 0
when to use l2(ridge)?
*when all the feature columns are important
*when multi colaranity problem occurs
elastic->l1+l2
it is regularization penalty which uses both l1 and l2
it selects important feature and also handles multi colaranity problem

what is gradient descent?
it is an optimization technique which is use to find the best weight
where we use gradient descent and ml?
*logistic regression internally uses gradient descent to find the best weight
*xtrain gradient boosting algorithm(xt boost) uses gradient descent find the best weight
*regularization technique uses gradient descent to find the best weight and reduces the variance

what is generalization?(imp)
how well the model captures the pattern of unseen data then the model is generalized well
if model fails to captures unseen data pattern then the model is not generalized well(overfit)
solution:regularization

----------------------------------------------------------------------------------------------------
classification matrics:
1)confussion matrix
2)precision score
3)Recall score
4)f1-score
5)Accuracy score

1)confussion matrix:
it is a classification matrix which tells out of all the samples how many samples are correctly and wrongly predicted from the class labels
ex:
output labels                 |                Predicted output |     
total=1000                    |                0              1 |             100  |  200
positive=700                  | Actual   0     TN            FP |            ------|-------
negative=300                  | output   1     FN            TP |             500  |   200
CONCLUSION: model performance is bad false negative and false positive are more than other two
ex2:
total samples=500
spam=400
not spam=100
80  |  20
----|-----
100 | 300
CONCLUSION: model is good
syntax:
from sklearn.metrics import confussion_matrix
confussion_matrix(Actual output,predicted output)
train data: confussion_matrix(ytrain,ypred_train)
test data: confussion_matrix(ytest,ypred_test)

2)precision score:
it is a classification metrics which tells out of all the predicted values how much % values are correctly predicted
i) precision score for positive label: out of all the predicted positive labels how much % labels are correctly predicted as positive
formula: TP/(TP+FP)
total samples=500
spam=400
not spam=100
80  |  20
----|-----
100 | 300
conclusion1: out of all the predicted positive labels(400),50% samples are correctly predicted as positive
conclusion2: out of all the predicted spam labels(400),% samples are correctly predicted as spam
ii) precision score for negative label:out of all the predicted negative labels how much % labels are correctly predicted as negative
conclusion1: out of all the predicted negative labels(100),% samples are correctly predicted as negative
syntax:
from sklearn.metrics import precision_score
precision_score(Actual output,predicted output,positive_label=1)
conclusion1: 
conclusion2: 

3)Recall score
from sklearn.metrics import recall_score
recall score for positive labels: it is classification matrics which tells out of all the actual values how many values are correctly predicted
positive recall score=TP/(TP+FN)
recall score for negative labels: out of all the actual negative labels how many labels are correctly predicted as negative
negative recall score=TN/(TN+FP)
                0          1
  700      0   TN         FP
  100      1   FN         TP

5)Accuracy score:
from sklearn.metrics import accuracy_score
it is classification metrics which tells out of all the samples how many samples are correctly predicted
Accuracy score=TN+TP/(TN+FN+TP+FP)
it is only used for balanced dataset not the imbalanced dataset
why not use Accuracy score with ex
                    0          1
1000         0     1000        0
200          1     200         0
it should calculating 83% as false Prediction so not used for imbalanced dataset

4)f1-score:
from sklearn.metrics import f1_score
it is a classification metrics which balances both precision and recall score and we use this for imbalanced dataset
f1-Score for negative:2*precision negative*recall negative/(precision negative+recall negative)
f1-Score for positive:2*precision positive*recall positive/(precision positivee+recall positive)

---------------------------------------------------------------------------------------------
column transformer:                                                          
it is a ml technique which combines multiple preprocessing and executes parallely.                    
in column transformer we use only preprocessing techniques not algorithms.                         
syntax:                           
from sklearn.compose import columnTransformer                                                   
preprocessing = ColumnTransformer(                                               
    transformer = [                                                        
        ('name of preprocessing',preprocessing,col_names)                                                   
        (--------------------||-------------------------)                                        
    ]                                                                                          
    remainder = 'drop'/'passthrough'                    
)                   
example:                                                                                          
preprocessing = ColumnTransformer(                                    
    transformer = [                                       
        ('minmax_scaling',MinMaxScalar(),                                  
        ['f1','f2']),                                       
        ('Robust_scaling',RobustScaler(),['f3','f4']),                              
        ('encoder',TargetEncoder(),['f5,'f6'])                                    
    ]                                              
)             
----------------------------------------------------------------------------------------
Pipeline:
it combines multiple steps like preproccesing and algorithm
it executes the steps one by one 
from sklearn.Pipeline import Pipeline
main_pipeline=Pipeline(
  steps=[
    ('name','steps')
    .
    .
  ])
example1:
main_pipeline=Pipeline(
  steps=[
    ('scaling','Standard Scales()')
    ('model','DecisionTreeRegressor())
  ])
main_pipeline.fit(Xtrain,ytrain)
example2:
preprocessing = ColumnTransformer(                                    
    transformer = [                                       
        ('Scaling',StadardScalar(),num_cols),                                                                     
        ('encoder',OneHotEncoder(handle_unknown='ignore'),cat_cols)                                    
    ])
note: sholud not use single same columns for multiple preproccesing techniques it will affect to model accuracy
example3:
preprocessing = ColumnTransformer(                                    
    transformer = [                                       
        ('nums_pre',Pipeline(
  steps=[
    ('scaling','Standard Scales()')
    ('poly','PolynomialFeatures()')]),num_cols),                                                                     
        ('encoder',OneHotEncoder(handle_unknown='ignore'),cat_cols)                                    
    ])
---------------------------------------------------------------------------------------------------
Ensamble
1)Bagging
2)Boosting
Ensamble: it is ml technique which uses multiple algorithm to improve the model performance
when we use this?
*when decision tree is not performing well then we use Ensamble technique
Bagging: it is an Ensamble technique which uses 100 decision tree by default and paralelly it preprocess all the trees 
Bagging we use when decision tree becomes overfit
Bagging reduces the variance not bias
what is Random Forest?
it is supervised learning algorithm which comes under Bagging Ensamble technique and we can use for both regression and classification
how Random Forest works?
*it creates 100 decision trees 
*creates random samples for all 100 trees with the help of bootstrapping sampling technique
*paralelly it trains all 100 trees
*all 100 trees are independent
max_samples=0.8
how random forest predicts the output?
for classification: *it takes majority
for regression: *average of all decision trees 
When we use Random Forest?
*decision tree is overfit use the Random Forest
when not to use Random Forest?
*decision tree is underfit not use the Random Forest
*larger dataset sholud not use Random Forest
syntax:
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
model=RandomForestClassifier(n_estimates=100)
---------------------------------------------------------------------------------------
what is boosting?
it is an ensemble technique which uses multiple decision tree which is use for both regression and classification
we use boosting when decision tree becomes underfit(it reduces to bias)
sequentially it process all 100 trees 
XGBoost:
it stands for extereme gradient boosting algorithm
which comes under boosting Ensemble technique 
techniques which is used internally:
* regularization
* gradient descent
Advantages of XGBoost:
1)scaling technique not required
2)it handles machine values
3)it handles categorical columns
4)it works well with large dataset
5)it handles both high variance and high bias

Sigmoid for binary
softmax for multi class
diff bw Bagging and boosting(imp)

----------------------------------------------------------------------------------------------------
KNN(K Nearest Neighbour)
it is supervised algorithm which is use for both classification algorithm
it uses distance formula to predicts the output
it uses 2 distance formula such as Euclidean, Manhattan
it is lazy algorithm, during model training just choose the data
Euclidean formula = sqrt((x2-x1)^2 + (y2-y1)^2)
feature1   |   feature2    |   Target
1          |     100       |    100
7          |      50       |    200
15         |      25       |    300
8          |      25       |    400
9          |      10       |    500
Euclidean for rows = 30,27,50
if the data is complex and high dimension then we use manhattan
Manhattan formula = |x2-x2|+|y2-y1|
----------------------------------------------------------------------------------------------------
Grid Search CV 
it is hyper parameter unique technique which balances both bias and variance by selecting the best parameter values
syntax:
from sklearn.model_selection import GridSearchCV
gridsearchcv=GridSearchCV(
  estimator=DecissionTreeClassifier()/pipeline
  cv=3|5|10
  parameter_grid={
    'parameter_name':[value1,value2.......],
    'parameter_value':[value1,value2.........]
  }
  n_jobs=1,2,3,4.........
  verboise=0,1,2,3,4.......
  score="r2"|"f1"|"accuracy"
)

scenario-1:
grid_search_cv=GridSearchCV(
  estimator=DecissionTreeClassifier(),
  param_grid={max_depth:[None,10,50,100],
            criterian:['gini','entropy']}
  cv=5)
grid_search_cv.fit(xtrain,ytrain)

scenario-2:
pipeline=Pipeline(
    steps=[
        ('preprocessing',preprocessing),
        ('model',DecisionTreeClassifier())
    ]
)
grid_search_cv=GridSearchCV(
  estimator=pipeline,
  param_grid={model__max_depth:[None,10,50,100]}
  cv=5)
grid_search_cv.fit(xtrain,ytrain)

5-Folds (equally Split)
Parameter Combinations & Results:
1) None, gini \implies \begin{cases} CV1 \implies 90\% \\ CV2 \implies 80\% \\ CV3 \implies 75\% \\ CV4 \implies 70\% \\ CV5 \implies 90\% \end{cases} \implies 81\% \ 6
2) None, entropy
3) 10, gini \implies 87\% \ 5
4) 10, entropy \implies 95\% \ 2
5) 50, gini \implies 90\% \ 4
6) 50, entropy \implies 71\% \ 7
7) 100, gini \implies 95.7\% \ 1
8) 100, entropy \implies 91\% \ 3

n_jobs=-1,1,2,3,4 
verbouse->it can used for getting the information from gridsearchcv
verbouse=0 -> nothing
verbouse=1 -> some information
verbouse=2 -> more information
cv5 -> default
cv3 -> large dataset
cv10 -> smaller dataset

import OS
print(Os.cpu_count())
----------------------------------------------------------------------------------------------------
bias variance trade-off
it means how we are balancing bias and variance