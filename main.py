import custom_emotion_analysis
print("Starting software for emotion-based image generation...")

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