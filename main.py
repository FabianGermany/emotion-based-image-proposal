#Import packages
import emotion_recognition.custom_emotion_analysis as custom_emotion_analysis
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import cv2

#import python files from other folders (GAN folder)
#sys.path.insert(0, 'GAN')
from GAN import inference
os.makedirs("GAN/output_training", exist_ok=True)

# Choose the reaction mode
#reaction_mode = "GAN_normal" #GAN image generation normal mode
#reaction_mode = "GAN_WoZ" #GAN image generation - Wizard-of-Oz mode
reaction_mode = "color" #Color mode

# Choose the operation mode
#operation_mode = "default"
operation_mode = "user_study"

# Function to display the default image
def display_default_image():
    default_image = cv2.imread("assets/Default_Img_MB.png")
    cv2.imshow('Co-Driver Display', default_image)

# Function to display the GAN image
def display_GAN_landscape_image():
    current_GAN_image = cv2.imread("GAN/output_inference/inference.png")
    cv2.imshow('Co-Driver Display', current_GAN_image)

# Function to display a real landscape image
def display_GAN_WoZ_image():
    WoZ_GAN_image = cv2.imread("assets/prototype/example_images/austria_1.jpg")
    cv2.imshow('Co-Driver Display', WoZ_GAN_image)

# Function to display one single color
def display_single_color():
    single_color_image = cv2.imread("assets/one-color.png")
    cv2.imshow('Co-Driver Display', single_color_image)

# Emotion detection function
#The normal way is to use DeepFace.stream or DeepFace.realtime.analysis for emotion recognition
#However, I need to access the values from DeepFace.stream, so I use my custom function stored in custom_emotion_analysis.py:
def emotion_recognizer(mode):
    return custom_emotion_analysis.custom_emotion_analyzer(
                               mode = mode,
                               db_path = '',
                               model_name ='VGG-Face',
                               detector_backend = 'opencv',
                               distance_metric = 'cosine',
                               enable_face_analysis = True,
                               source = 0, #0=webcam
                               time_threshold = 3, #how many seconds the analyzed display will be displayed; must be larger than 1
                               frame_threshold = 10 #amount of frames to focus on face; must be larger than 1
                               )

# Procedure
print("Starting software for emotion-based image generation...")
cv2.namedWindow('Co-Driver Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Co-Driver Display', 800, 500)
cv2.moveWindow('Co-Driver Display', 700, 20)
display_default_image() #print default image
while(True):
    inference.generate_new_GAN_img(model_path="GAN/models/existing_generator.pth", output_path="GAN/output_inference/inference.png")  # generate GAN image for the next time proactively
    status = emotion_recognizer(mode=operation_mode) #run emotion recognition script
    if (status == "bad emotion"):
        if (reaction_mode == "GAN_normal"):
            display_GAN_landscape_image() #propose a soothing picture via GAN
        elif (reaction_mode == "GAN_WoZ"):
            display_GAN_WoZ_image()
        elif (reaction_mode == "color"):
            display_single_color()
    elif (status == "good emotion"):
        display_default_image()  # propose a soothing picture via GAN