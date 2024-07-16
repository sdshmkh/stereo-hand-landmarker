from datetime import datetime, timedelta
import numpy as np
import cv2 as cv
import mediapipe as mp
from typing import *

from parallelism import RedisProducer, RedisEncoder, RedisConsumer, SyncRedisProducer
from projection import project3d
from visualization import Visualizer


RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

def configure_mp_options(model_path: str, running_mode=RunningMode.LIVE_STREAM, result_callback=lambda x, y, z: print("Default:", x)):
    # configure pose landmarker
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=running_mode,
    result_callback=result_callback
    )
    return options


class RedisHandlandMarkerResult(RedisEncoder):
    def __init__(self, cam_id:int, timestamp:int = 0, result:HandLandmarkerResult = None, terminate:bool = False) -> None:
        self.result = result
        self.timestamp = timestamp
        self.terminate = terminate
        self.cam_id = cam_id

class Landmarker3D(RedisEncoder):
    def __init__(self, points, terminate=False) -> None:
        super().__init__()
        self.points = points
        self.terminate = terminate

    
class Landmarker():
    def __init__(self, cam_id, dir="calibration") -> None:
        self.model_path = "model/handlandmarker_models/hand_landmarker.task"
        self.cap: cv.VideoCapture = cv.VideoCapture(cam_id)
        self.cam_id = cam_id
        print(self.cam_id)
        
        options = configure_mp_options(self.model_path, result_callback=self.handle_result)
        self.landmarker = HandLandmarker.create_from_options(options)
        print("Lander", self.landmarker)

        # load rectification maps
        self.remap_x = None
        self.remap_y = None
        remaps = np.load("{}/extrinsics/stereo-rectified-maps.npz".format(dir))
        if self.cam_id == 0:
            self.remap_x, self.remap_y = remaps['left_map_x'], remaps['left_map_y']
        elif self.cam_id == 1:
            self.remap_x, self.remap_y = remaps['right_map_x'], remaps['right_map_y']
            


    def rectify_image(self, frame):
        return frame
        r_frame = cv.remap(frame, self.remap_x, self.remap_y, cv.INTER_LANCZOS4)
        cv.imwrite("recitfied_frame_{}.png".format(self.cam_id), r_frame)
        return r_frame

    def handle_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        pass
        
    def detect_landmarks(self) -> bool:
        if not self.cap.isOpened():
            self.cap.release()
            raise Exception("Could not capture video")
        start = datetime.now()
        print("Started Detecting...", start)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = (datetime.now() - start).total_seconds() * 1e3
            mp_timestamp = mp.Timestamp.from_seconds(timestamp).value

            # recitfy frame
            rectified_frame = self.rectify_image(frame)

            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rectified_frame)
            self.landmarker.detect_async(mp_frame, mp_timestamp)

            # cv.imshow("Cam: {}".format(self.cam_id), frame)
            # if timestamp > 2*1e4:
            #     break
        self.cap.release()
        cv.destroyAllWindows()
        return False


class RedisLandmarker(Landmarker, RedisProducer):
    def __init__(self, channel: str, cam_id: int) -> None:
        Landmarker.__init__(self, cam_id)
        RedisProducer.__init__(self, channel)
    
    def handle_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        if len(result.hand_landmarks) > 0:
            print("Detect in {}".format(self.cam_id))
        extended_result = RedisHandlandMarkerResult(self.cam_id, timestamp_ms, result)
        self.produce(extended_result)
    
    def run(self) -> None:
        self.detect_landmarks()
        terminate = RedisHandlandMarkerResult(self.cam_id, terminate=True)
        self.produce(terminate)


class StereoLandmarker(RedisConsumer, SyncRedisProducer):
    def __init__(self, ) -> None:
        RedisConsumer.__init__(self, ["channel_cam_0", "channel_cam_1"])
        SyncRedisProducer.__init__(self, "channel_points_3d")
    
    def consume(self, message_objs: List[RedisHandlandMarkerResult]) -> bool:
        # check for termination condition
        terminate = any([message_obj.terminate for message_obj in message_objs])
        if terminate:
            self.produce(Landmarker3D([], terminate=True))
            return False

        # check if landmark have been detected in both frames
        landmark_check = [check_for_landmarks(message_obj) for message_obj in message_objs]
        print(landmark_check)
        if not all(landmark_check):
            # keep listening
            print("exit as no landmark detected")
            return True
        print("Landmark detected...", message_objs[0].timestamp, message_objs[1].timestamp)

        # get points for each camera
        landmarks = landmarks_to_numpy(message_objs)
        print("landmark shape", landmarks.shape)
        # get 3d points for each camera
        points3d = project3d(landmarks[:, 0], landmarks[:, 1], "calibration")
        print("points 3d shape", points3d.shape)
        l3d = Landmarker3D(points3d.tolist())

        # push them into a pub/sub queue
        self.produce(l3d)
        return True


def landmarks_to_numpy(message_objs: List[RedisHandlandMarkerResult]):
    # 21 landmarks - mediapipe
    all_landmarks = list()
    for landmark_id in range(21):
        curr_landmark = list()
        for message in message_objs:
            landmark = message.result.hand_landmarks[0][landmark_id]
            normalized_coordinates = normalized_to_pixel_coordinates(landmark.x, landmark.y, 1920, 1080)
            curr_landmark.append(normalized_coordinates)
        all_landmarks.append(curr_landmark)
    as_numpy = np.array(all_landmarks)
    return as_numpy

def check_for_landmarks(message: RedisHandlandMarkerResult) -> bool:
    landmarks = message.result.hand_landmarks
    return len(landmarks) > 0


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or np.isclose(0, value)) and (value < 1 or
                                                      np.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    print("*"*50)
    return -1, -1
  x_px = min(np.floor(normalized_x * image_width), image_width - 1)
  y_px = min(np.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px