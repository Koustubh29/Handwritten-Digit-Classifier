import streamlit as st
import numpy as np
from pickle import load
from PIL import Image
import sklearn

st.title('Image Detection')
classifier = load(open(r'C:/Users/HP\Data Science with Python/01 Python Programming/Machine Learning/models/rf_model.pkl', 'rb'))
uploaded_file = st.file_uploader("Choose a file",type=["png","jpg"])
if uploaded_file is not None:
    st.image(uploaded_file, width=250)
    img = Image.open(uploaded_file)
    array = np.array(img.resize((28,28)))
    df = array[:,:,0].flatten()
    new = df.reshape(1,-1)

button = st.button("predict")
if button == True:
    prediction = classifier.predict(new)
    st.text('The number is: ')
    st.title(prediction[0])