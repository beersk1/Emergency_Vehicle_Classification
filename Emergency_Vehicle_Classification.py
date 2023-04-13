import pandas as pd 
import numpy as np 
import streamlit as st
import boto3
import tempfile
from joblib import load
import tensorflow 
import keras
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from secret import access_key, secret_access_key
import requests
from streamlit_lottie import st_lottie_spinner


st.title('Emergency Vehicle Prediction')




st.write("""## Upload Vehicle Images for prediction""")
image = st.file_uploader("")
predict_bt = st.button('Predict')


# id
# AKIAXD7Y37IDU4IA2XR7
# key
# lIENMF0V630oz4r06GdNdtyyNfhmdV+ug5+0nfnt



def make_prediction():
    # connect to s3 bucket with the access and secret access key
    client = boto3.client(
        's3', aws_access_key_id=st.secrets["access_key"],aws_secret_access_key=st.secrets["secret_access_key"],region_name='ap-south-1')
    
    
    bucket_name = "emergencyprediction"
    key = "emergency.h5"

    # load the model from s3 in a temporary file
    with tempfile.NamedTemporaryFile() as fp:
        # download our model from AWS
        client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)

        # load the model using Keras library
        model = load_model(fp.name)

    # prediction from the model, returns between 0 and 1 
    return model.predict(test_image) 


@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_loading_an = load_lottieurl(
    'https://assets6.lottiefiles.com/packages/lf20_t9gkkhz4.json')


if predict_bt:


    with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
       
        test_image = load_img(image, target_size = (150,150)) #formobilenet
        test_image = img_to_array(test_image)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        final = make_prediction()
        # print(result)
        st.image(image)
        if final > 0.5:
            st.success("Normal Vechicle")
        else:
            st.success("Emergency Vechicle")
            


