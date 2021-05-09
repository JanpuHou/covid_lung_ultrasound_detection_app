import streamlit as st
from image_classification import teachable_machine_classification
from PIL import Image, ImageOps
import numpy as np



st.title("Lung Ultrasound Image Classification")
st.header("Covid ?")
st.text("Upload a Lung Ultrasound Image for image classification as Covid, Healthy or Pneumonia")

uploaded_file = st.file_uploader("Choose a Lung Ultrasound ...", type="jpg","png")

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
