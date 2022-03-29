import custom_emotion_analysis
import matplotlib.pyplot as plt
import matplotlib.image as image
import time
import os

#import python files from other folders (GAN folder)
import sys
#sys.path.insert(0, 'GAN')
#from GAN import inference
from GAN import inference

os.makedirs("GAN/output_training", exist_ok=True)


# Define settings and for image displaying
#------------------------
default_image = image.imread("assets/Default_Img_MB.png")
plt.axis("off")
#plt.plot(1)
plt.ion()
plt.show()

# Function to display the default image
#------------------------
def display_default_image():
    plt.imshow(default_image)
    plt.pause(0.001)
    #plt.draw()
    #plt.show(block=False)


def display_GAN_landscape_image():
    current_GAN_image = image.imread("GAN/output_inference/inference.png")
    plt.imshow(current_GAN_image)
    plt.pause(0.001)
    # plt.draw()
    # plt.show(block=False)

print("Starting software for emotion-based image generation...")




# Procedure
# ------------------------
inference.generate_new_GAN_img(model_path="GAN/models/existing_generator.pth", output_path="GAN/output_inference/inference.png") # generate GAN image for the next time proactively #todo
display_default_image() #print default image
time.sleep(3)
display_GAN_landscape_image() #show a GAN image

#todo if bedingung hier reinsourcen
#The normal way is to use DeepFace.stream or DeepFace.realtime.analysis for emotion recognition
#However, I need to access the values from DeepFace.stream, so I use my custom function stored in custom_emotion_analysis.py:
custom_emotion_analysis.custom_emotion_analyzer(db_path = '',
                           model_name ='VGG-Face',
                           detector_backend = 'opencv',
                           distance_metric = 'cosine',
                           enable_face_analysis = True,
                           source = 0, #0=webcam
                           time_threshold = 3, #how many seconds the analyzed display will be displayed; must be larger than 1
                           frame_threshold = 10 #amount of frames to focus on face; must be larger than 1
                           )
#this function includes the emotion_analysis including the reactions to it (displaying pictures)


plt.show()