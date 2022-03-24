#!pip install DeepFace
from deepface import DeepFace
print("Starting software...")

#start emotion recognition camera livestream
# DeepFace.stream(db_path = '',
#                 model_name ='VGG-Face',
#                 detector_backend = 'opencv',
#                 distance_metric = 'cosine',
#                 enable_face_analysis = True,
#                 source = 0, #0=webcam
#                 time_threshold = 3, #how many seconds the analyzed display will be displayed
#                 frame_threshold = 10 #amount of frames to focus on face
#                 )

#there are easy ways by using DeepFace.stream. However, I need to access the values from DeepFace.stream, so I write my custom function:
DeepFace.realtime.analysis(db_path = '',
                           model_name ='VGG-Face',
                           detector_backend = 'opencv',
                           distance_metric = 'cosine',
                           enable_face_analysis = True,
                           source = 0, #0=webcam
                           time_threshold = 3, #how many seconds the analyzed display will be displayed; must be larger than 1
                           frame_threshold = 10 #amount of frames to focus on face; must be larger than 1
                           )

#todo here ringbuffer etc.

#access values and check for too low valence or too high or too low arousal

#propose a soothing picture via GAN