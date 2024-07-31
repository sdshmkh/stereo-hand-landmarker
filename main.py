from model import StereoLandmarker, RedisLandmarker
from visualization import StreamingVisualizer


# Camera detectors
rlh = RedisLandmarker(["channel_cam_0", "world_landmarks_cam_0"], cam_id=0)
rlh2 = RedisLandmarker("channel_cam_1", cam_id=1)


# Consumer
stereo = StereoLandmarker()

stereo.start()
rlh.start()
rlh2.start()
print("started")
stereo.join()
rlh.join()
rlh2.join()