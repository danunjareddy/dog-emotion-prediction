import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Load your .keras model
@st.cache_resource
def load_my_model():
    # Make sure the filename matches exactly what you downloaded
    return tf.keras.models.load_model('dog_emotion_model.keras')

model = load_my_model()

# 2. Define your labels (Make sure these are in the same order as your training)
class_names = ['Angry', 'Happy', 'Relaxed', 'Sad'] 

st.set_page_config(page_title="Dog Emotion Detector", page_icon="🐶")
st.title("🐶 Dog Emotion Prediction")
st.write("Upload a photo of a dog, and the AI will tell you how it feels!")

# 3. File Uploader
file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Display the image
    image = Image.open(file)
    st.image(image, caption='Uploaded Dog Image', use_container_width=True)
    
    # 4. Preprocessing
    # IMPORTANT: Use the same size you used in Colab (usually 224x224 or 150x150)
    size = (224, 224) 
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # If your model expects 0-1 range (normalization):
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # 5. Prediction
    with st.spinner('Analyzing emotion...'):
        prediction = model.predict(img_reshape)
        # Get the index of the highest probability
        result_index = np.argmax(prediction[0])
        label = class_names[result_index]
        confidence = prediction[0][result_index] * 100

    # 6. Show Result
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2f}%")