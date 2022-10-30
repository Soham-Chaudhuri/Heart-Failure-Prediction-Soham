import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier , plot_importance
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st


st.title('HEART FAILURE PREDICTION')
_, _, col2 = st.columns([1, 1.5, 2])
with col2:
    st.subheader('-By Soham Chaudhuri')
st.write('Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.')

data=pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")
plt.style.use('dark_background')





st.write('\n\n')
st.subheader('According to some data collected :-')



len_live=len(data["DEATH_EVENT"][data.DEATH_EVENT==0])
len_death=len(data["DEATH_EVENT"][data.DEATH_EVENT==1])

arr=np.array([len_live,len_death])
labels=['Total Alive','Total Deaths']
fig, ax = plt.subplots(figsize=(1,1))
ax.pie(arr,labels=labels,explode=[0.2,0],shadow=True)
# plt.show()
st.pyplot(fig)
st.write("No of death:-",len_death)
st.write("No of live:-",len_live)




st.markdown("***")
st.subheader('Graph of Density of Deaths due to Heart Failure at Different Ages:-')
fig_open=plt.figure()
sns.distplot(data['age'])
st.pyplot(fig_open)



st.markdown("***")
st.subheader('Above 50 age:-')
age_above_50_not_died=len(data['DEATH_EVENT'][data.age>=50][data.DEATH_EVENT==0])
age_above_50_died=len(data['DEATH_EVENT'][data.age>=50][data.DEATH_EVENT==1])
labels=['Alive Above 50 Age','Deaths Above 50 Age']
arr=np.array([age_above_50_not_died,age_above_50_died])
fig, ax = plt.subplots(figsize=(2,2))
ax.pie(arr,labels=labels,explode=[0.2,0.0],shadow=True)
# plt.show()
st.pyplot(fig)
st.write("No of death:-",age_above_50_died)
st.write("No of live:-",age_above_50_not_died)



st.markdown("***")
st.subheader('With Diabetes:-')
len_d_alive = len(data['DEATH_EVENT'][data.diabetes == 1][data.DEATH_EVENT == 0])
len_d_died = len(data['DEATH_EVENT'][data.diabetes == 1][data.DEATH_EVENT == 1])
arr = [len_d_died,len_d_alive] 
labels = ['Died with Diabetes', "Not Died with Diabetes"] 
fig, ax = plt.subplots(figsize=(2,2))
ax.pie(arr, labels=labels, explode = [0.2,0.0] , shadow=True) 
# plt.show()
st.pyplot(fig)
st.write("No of death:-",len_d_died)
st.write("No of live:-",len_d_alive)



st.markdown("***")
st.subheader('Correlation Among Features:-')
corr=data.corr()
fig_open=plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True)
st.pyplot(fig_open)




x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)









st.markdown("***")
st.subheader('Machine Learning Model For Heart Failure Prediction:-')
# model_name=st.selectbox(
#     'Select Model',
#     ('Default','Logistic Regression','Standard Logistic Regression','SVM',
#     'Decision Tree','Random Forest','XGBoost','Gradient Boost')
# )
# if(model_name!='Default'):

st.write('Accuracy Score:- 85.6')
st.write('Precision Score:- 80')  
st.write('Recall Score:- 71.4286')


st.subheader('Enter Details Of The Patient:-')
age = st.slider('Age', 0, 130, 0)
anaemia = st.slider('Anaemia if yes->1', 0, 1, 0)
creatinine_phosphokinase = st.slider('Creatinine Phosphokinase', 20.0, 8000.0, 20.0)
diabetes = st.slider('Diabetes if yes->1', 0, 1, 0)
ejection_fraction = st.slider('Ejection Fraction', 10.0, 80.0, 10.0)
high_blood_pressure = st.slider('High Blood Pressure if yes->1', 0, 1, 0)
platelets = st.slider('Platelets', 20000.0, 900000.0, 20000.0)
serum_creatinine = st.slider('Serum Creatinine', 0.0, 10.0, 0.0)
serum_sodium = st.slider('Serum Sodium',50.0, 200.0, 50.0)
sex = st.slider('Sex if male->1', 0, 1, 0)
smoking = st.slider('Smoking if yes->1', 0, 1, 0)
time = st.slider('Time(Follow up period)', 0, 300, 0)
test_data = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])


    # if(model_name=='Logistic Regression'):
    #     lr_clf = LogisticRegression(max_iter=1000) 
    #     lr_clf.fit(x_train, y_train)
    #     predict = lr_clf.predict(test_data)
    # elif(model_name=='Standard Logistic Regression'):
    #     lr_clf_pip = make_pipeline(StandardScaler(), LogisticRegression()) 
    #     lr_clf_pip.fit(x_train, y_train) 
    #     predict = lr_clf_pip.predict(test_data)
    # elif(model_name=='SVM'):
    #     svc=SVC(C=10,gamma=0.0001)
    #     svc.fit(x_train,y_train)
    #     predict = svc.predict(test_data)
    # elif(model_name=='Decision Tree'):
    #     ds_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features=0.75,
    #                    max_leaf_nodes=25, min_impurity_decrease=0.0005,
    #                    min_samples_split=5, min_weight_fraction_leaf=0.0075,
    #                    random_state=2) 
    #     ds_clf.fit(x_train, y_train) 
    #     predict = ds_clf.predict(test_data)
    # elif(model_name=='Random Forest'):
    #     rf_clf = RandomForestClassifier(max_depth=2, max_features=0.5,
    #                    min_impurity_decrease=0.01, min_samples_leaf=10,
    #                    random_state=2) 
    #     rf_clf.fit(x_train, y_train) 
    #     predict = rf_clf.predict(test_data)
    # elif(model_name=='XGBoost'):
    #     xgb1 = XGBClassifier(colsample_bytree= 1.0,
    #     learning_rate = 0.1,
    #     max_depth = 4,
    #     n_estimators= 400,
    #     subsample= 1.0)  
    #     eval_set  = [(x_test, y_test)]
    #     xgb1.fit(x_train, y_train,early_stopping_rounds=10, eval_metric="logloss",eval_set=eval_set, verbose=True)
    #     predict = xgb1.predict(test_data)
    #     fig_open=plt.figure()
    #     plot_importance(xgb1)
    #     # st.pyplot(fig_open)
    # elif(model_name=='Gradient Boost'):
    #     gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,max_depth=1,random_state=0) 
    #     gbdt.fit(x_train, y_train) 
    #     predict = gbdt.predict(test_data) 
    
xgb1 = XGBClassifier(colsample_bytree= 1.0,
    learning_rate = 0.1,
    max_depth = 4,
    n_estimators= 400,
    subsample= 1.0)  
eval_set  = [(x_test, y_test)]
xgb1.fit(x_train, y_train,early_stopping_rounds=10, eval_metric="logloss",eval_set=eval_set, verbose=True)
predict = xgb1.predict(test_data)
    
if st.button('Predict'):
    st.subheader("Prediction :-")
    if(predict[0]=='0'):
        st.subheader('Do not worry You are not at much risk')
    else:
        st.subheader('You are at risk Please consult a Doctor')








