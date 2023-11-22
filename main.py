import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

st.title("ML Mini-Project")

classifier_name = st.sidebar.selectbox("Select Classifier", ("Decision Tree", "SVM", "Random Forest"))
kernel_user = 'linear'
if classifier_name == 'SVM':
    kernel_user = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))

# inputs for prediction
N = st.sidebar.slider('K', 1, 140)
P = st.sidebar.slider('N', 5, 145)
K = st.sidebar.slider('P', 5, 205)

temp = st.sidebar.number_input("Temperature", value=None, placeholder="Enter Value")
hum = st.sidebar.number_input("Humidity", value=None, placeholder="Enter Value")
pH = st.sidebar.number_input("pH Value", value=None, placeholder="Enter Value")
rain = st.sidebar.number_input("Rainfall", value=None, placeholder="Enter Value")

vals = [N, P, K, temp, hum, pH, rain]

# Read from csv
data = pd.read_csv("Crop_recommendation.csv")
X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
y = data.iloc[:, [-1]]
encode = dict(data["label"].value_counts())
before = y[:]

# Encode the categorical target variable
le = LabelEncoder()
y = le.fit_transform(y.values.ravel())

for i in range(len(y)):
    encode[before.iat[i, 0]] = y[i]

decode = {}
for key, val in encode.items():
    decode[val] = key

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt_model.fit(X_train, y_train)

vals = np.array(vals).reshape(1, -1)

col1, col2 = st.columns(2)

# Reset Values button
if col2.button("Reset Values", type="primary"):
    vals = []

# Classify button
if col1.button('Classify'):
    if classifier_name == 'Decision Tree':
        y_pred_dt = dt_model.predict(X_test)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        st.title(f'Result using Decision Tree = {decode[dt_model.predict(vals)[0]]}')
        st.subheader(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
        st.subheader("Decision Tree Visualization")
        fig = plt.figure(figsize=(20, 10))
        plot_tree(dt_model, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
        st.pyplot(fig)
    elif classifier_name == 'SVM':
        svm_model = SVC(kernel=kernel_user, random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        st.title(f"Result using {kernel_user} kernel = {decode[svm_model.predict(vals)[0]]}")
        st.subheader(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

        # plot
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        vals_pca = pca.transform(vals)
        svm_model = SVC(kernel=kernel_user, random_state=42)
        svm_model.fit(X_train_pca, y_train)
        fig = plt.figure(figsize=(10, 6))
        plot_decision_regions(X_train_pca, y_train, clf=svm_model, legend=2)
        plt.title(f'SVM Decision Regions (with PCA) - {kernel_user} kernel')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        st.pyplot(fig)
