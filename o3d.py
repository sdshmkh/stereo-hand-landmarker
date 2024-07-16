import open3d as o3d
import numpy as np

from visualization import Visualizer, StreamingVisualizer


def load_extrinsics():
    dir = "calibration"
    stereo_calibration = np.load("{}/extrinsics/stereo_calibration.npz".format(dir))
    return [stereo_calibration["R"]], [stereo_calibration["T"]]


r, t = load_extrinsics()

print(r, t)

sv = StreamingVisualizer(cams=2, R=r, T=t)

sv.consume()
# sv.display_scene()

# sv.viz.run()
