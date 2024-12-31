import mediapipe as mp, os
from mediapipe import tasks
from collections import namedtuple
from mediapipe.tasks.python import vision
model_path = os.path.join('Data', 'Models', 'face detector', 'blaze_face_short_range.tflite') #path to the face detector

def get_boundry_box(numpy_image , cropping_width=451 , cropping_hight=300):
    BaseOptions = tasks.BaseOptions
    FaceDetector = vision.FaceDetector
    FaceDetectorOptions = vision.FaceDetectorOptions
    VisionRunningMode = vision.RunningMode
    image_height , image_width , _ = numpy_image.shape
    x = (image_width  - cropping_width) // 2
    y = (image_height  - cropping_hight) // 2
    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    with FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        face_detector_result = detector.detect(mp_image)
    if face_detector_result.detections == []:
        boundry_box = namedtuple("BoundingBox" , ["origin_x" , "origin_y" , "width" , "height"])
        return boundry_box(x , y , cropping_width , cropping_hight)
    return face_detector_result.detections[0].bounding_box
