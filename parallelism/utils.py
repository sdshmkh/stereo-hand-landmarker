import numpy as np

def landmarks_to_numpy(message_objs):
    landmarks = list()
    for obj in message_objs:
        l_obj = list()
        for landmark in obj.result.landmarks:
            l_obj.append((landmark.x, landmark.y, landmark.z))
    landmarks = np.vstack(landmarks)
    return landmarks