#import libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings


from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, auc, precision_score, classification_report, roc_auc_score, recall_score

from PIL import Image
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

#create a title and a subtitle
st.write("""
#Heart Disease Prediction
This program predicts Heart Disease using Machine Learning
""")

#open and display an image
image = Image.open('C:/Users/hp user/PycharmProjects/heartwebapp/heart disease prediction.png')
st.image(image, caption = 'ML', use_column_width=True)

#import data
heart_data = pd.read_csv('C:/Users/hp user/PycharmProjects/heartwebapp/heart.csv')

#set a subheader
st.subheader('Data Information: ')
#show the data as a table
st.dataframe(heart_data)

'''
#show statistics
st.write(heart_data.describe)
#show the chart
chart = st.bar_chart(heart_data)
'''

#split data
features = heart_data.drop(columns = 'target', axis = '1')
target = heart_data['target']
features_train, features_test, target_train, target_test = train_test_split(features, target)

#scale data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
print("Features have been scaled")

#define classifiers
lr = LogisticRegression(max_iter=2000)
rf = RandomForestClassifier(n_estimators=3000)
svm = SVC(probability=True, kernel='linear', C=2, gamma='auto')

def get_model():
    model_options = np.array(['LR', 'RF', 'SVM'])
    model_input = st.selectbox('Model', model_options)

    if model_input == 'LR':
        algorithm = lr
    elif model_input == 'RF':
        algorithm = rf
    else:
        algorithm = svm

    return algorithm
def get_user_input():
    age = st.slider('Age', 0, 100, 50)
    sex = st.selectbox('Gender', ('1 - Male', '0 - Female'))
    cp = st.selectbox('Does patient have chest pain?', ('1 - typical angina', '2 - atypical angina', '3 - non-anginal pain', '4 - asymptomatic'))
    trestbps = st.slider('Patient resting blood pressure (in mm Hg on admission to the hospital)', 50, 250, 50)
    chol = st.slider('Patient serum cholestoral in mg/dl', 100, 600, 150)
    fbs = st.slider('Patient fasting blood sugar > 120 mg/dl? Select 0 for False and 1 for True', 0, 1, 1)
    restecg = st.selectbox('Resting electrocardiographic results', ('0 - Normal', '1 - having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 'showing probable or definite left ventricular hypertrophy by Estes criteria'))
    thalach = st.slider('Patient maximum heart rate achieved', 50, 250, 180)
    exang = st.selectbox('Does patient have exercise induced angina?', ('1 - Yes', '0 - No'))
    oldpeak = st.slider('ST depression induced by exercise relative to rest', min_value = 0.0, max_value = 10.0, value = 5.0, step= 0.1)
    slope = st.selectbox('Slope of the peak exercise ST segment', ('1 - Upsloping', '2 - Flat', '3 - Downsloping'))
    ca = st.slider('Number of major vessels (0-3) colored by flourosopy', 0, 3, 1)
    thal = st.selectbox('Patient history with thalassemia', ('1 - Normal', '2 - Fixed defect', '3 - Reversable defect'))

    #change values to data format
    #sex
    if sex == "1 - Male":
        sex = 1
    else:
        sex = 0

    #cp
    if cp == "1 - typical angina":
        cp = 1
    elif cp == "2 - atypical angina":
        cp = 2
    else:
        cp = 3

    #fbs
    if restecg == "0 - Normal":
        restecg = 0
    else:
        restecg = 1

    #exang
    if exang == "1 - Yes":
        exang = 1
    else:
        exang = 0

    #slope
    if slope == "1 - Upsloping":
        slope = 1
    elif slope == "2 - Flat":
        slope = 2
    else:
        slope = 3

    #thal
    if thal == "1 - Normal":
        thal = 1
    elif thal == "2 - Fixed defect":
        thal = 2
    else:
        thal = 3

    user_data = {'age': age,
                 'sex': sex,
                 'cp': cp,
                 'trestbps': trestbps,
                 'chol': chol,
                 'fbs': fbs,
                 'restecg': restecg,
                 'thalach': thalach,
                 'exang': exang,
                 'oldpeak': oldpeak,
                 'slope': slope,
                 'ca': ca,
                 'thal': thal
                 }
    #transform the data to df
    features = pd.DataFrame(user_data, index = [0])
    return features



#store the user input into a variable
user_input = get_user_input()

#set a subheader
st.subheader('User Input: ')
st.write(user_input)

#fit model
user_input_model = get_model()
st.write(user_input_model)

user_input_model.fit(features_train, target_train)

st.subheader("Accuracy of the model: ")
np_target = np.array(target_test)
target_reshaped = np_target.reshape(-1,1)
features_train_prediction = user_input_model.predict(features_test)
accuracy = float(accuracy_score(features_train_prediction, target_test))

st.write(str(round(accuracy*100,2)) + '%')


#store predictions
prediction = user_input_model.predict(user_input)
st.header('Classification: ')
if prediction == 0:
    st.subheaderheader('The model has predicted that you do not have heart disease')
else:
    st.subheader('The model has predicted that you do have heart disease')



