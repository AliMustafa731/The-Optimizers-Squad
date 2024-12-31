This is the capstone project for the Samsung Innovation Campus AI course.
The project is about implementing a face verification system that is based on Siamese Neural Networks using CNNs and Vision Transformers.

To run the project, do the following:
1. Install the requirements.txt
2. Download the face detector model from: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
3. Download the siamese model weights from: https://drive.google.com/drive/folders/1VdQbUAgehkjQYpGzxUL9wdv0l5Exv7lz?usp=sharing
4. Put the face detector model in a "Data/Models/face detector" folder, or choose your own directory structure, but make the necessary changes in the face_detector.py file.
5. Put the siamese model weights in a "Data/Models/Fine tuned model using LFW" folder, or choose your own directory structure and make the necessary changes to the app.py file.
6. Run the app.py file and follow the link to your browser to see the application running!
