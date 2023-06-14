from imutils import paths

known_emotion = ("angry", "disgust", "scared", "happy", "sad", "surprised", "neutral")

for emotion in known_emotion:
     list_path = list(paths.list_images("dataset/"+emotion))
     print("nombre d'image pour emotion  : {} , {} images".format(emotion, len(list_path))) 