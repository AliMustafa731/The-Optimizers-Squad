import gradio as gr
from face_detector import get_boundry_box
from models import get_compiled_siamese_model
import numpy as np
import cv2, os
image_size = (128, 128)

siamese_model = get_compiled_siamese_model(
    weights_path = os.path.join('Data', 'Models', 'Fine tuned model using LFW', 'model.weights.h5') #path to the siamese model
)

def process_images(image_A , image_B , threshold):
    image_A = np.array(image_A)
    image_B = np.array(image_B)
    bounded_box_A = get_boundry_box(image_A)
    bounded_box_B = get_boundry_box(image_B)
    x1 , y1 , w1 , h1 = bounded_box_A.origin_x , bounded_box_A.origin_y , bounded_box_A.width , bounded_box_A.height
    x2 , y2 , w2 , h2 = bounded_box_B.origin_x , bounded_box_B.origin_y , bounded_box_B.width , bounded_box_B.height
    image_A = cv2.resize(image_A[y1:y1+h1 , x1:x1+w1], image_size)
    image_B = cv2.resize(image_B[y2:y2+h2 , x2:x2+w2], image_size)

    distance = siamese_model.predict(x = [np.expand_dims(image_A, axis = 0), np.expand_dims(image_B, axis = 0)])[0][0] 
    if distance < threshold:
        return 'Yes' , distance
    else:
        return 'No' , distance

demo = gr.Interface(
    fn=process_images,
    title="FACE VERIFICATION WITH CNN+ViT AND SIAMESE NETWORK",
    description="This is a face verification model which uses Convolutional Neural Network and Vision Transformer to extract features from the image and Siamese Network to compare the features.",
    theme="ocean",
    inputs=[gr.Image(height = 400, width = 400, label="ORIGINAL IMAGE"),gr.Image(height = 400, width = 400, label="IMAGE TO COMPARE WITH" , ) , gr.Slider(minimum=0.0, maximum=1.0, value = 0.5, label="Distance Threshold")],
    outputs=[gr.Textbox(scale = 5, label="IS IT THE SAME PERSON?") , gr.Textbox(scale = 5, label="DISTANCE BETWEEN THE IMAGES")],
    flagging_mode="never"
)
demo.launch(inbrowser=True,share=True)