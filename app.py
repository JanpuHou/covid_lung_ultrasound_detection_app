import streamlit as st
from image_classification import teachable_machine_classification
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
     page_title="DIGI+ Homework App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9298069"
     }
 )
st.title("AI-Assisted Hand-held Covid Lung Ultrasound Image Screener")

image = Image.open('app_usage.jpg')
# st.image(image, caption='How This App Interpret Your Lung Ultrasound Image')
st.image(image,"Upload a Lung Ultrasound Image for image classification as Healthy, Covid or Pneumonia" )

uploaded_file = st.file_uploader("Choose a Lung Ultrasound ...", type="jpg")

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Lung Ultrasound', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	label = teachable_machine_classification(image, 'covid_classification.h5')
	if label == 0:
		st.write("The Ultrasound scan is Covid")
	elif label == 1:
		st.write("The Ultrasound scan is Healthy")
	else:
		st.write("The Ultrasound scan is Pneumonia")	

