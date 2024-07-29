
import open3d as o3d
import numpy as np
from branca.colormap import linear
import mediapipe as mp

from parallelism import SyncRedisConsumer

Connections = mp.tasks.vision.HandLandmarksConnections

class Visualizer():
    def __init__(self, cams=2, R=[], T=[]):
        self.cams = cams
        self.R = R
        self.T = T
        
        self.camera_positions = [np.array([0, 0, 0], dtype=np.float32)]
        origin = np.array([0, 0, 0])
        for r, t in zip(self.R, self.T):
            t = t.flatten()
            camera_pos = -1 * t.copy()
            self.camera_positions.append(camera_pos)
        
        print(self.camera_positions)
        self.viz: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self.hands = HandLineset()


    def display_scene(self):
        "creates an Open3D scene with cameras"
        # convert camera coordinates to point clouds
        camera_pcd = o3d.geometry.PointCloud()
        camera_pcd.points = o3d.utility.Vector3dVector(self.camera_positions)

        self.viz.create_window()

        self.viz.add_geometry(camera_pcd)
        colors = [(1, 0, 0), (0, 0, 1)]
        for idx, cam_pos in enumerate(self.camera_positions):
            cam_line_pcd = camera_lineset(cam_pos, 3, 3, color=colors[idx])
            self.viz.add_geometry(cam_line_pcd)
        
        self.viz.add_geometry(self.hands.hand_pcd)
        self.viz.add_geometry(self.hands.hand_lineset)


    def update_scene(self, landmarks):
        updated_hands_pcd, updated_hand_lineset = self.hands.update(landmarks)
        self.viz.update_geometry(updated_hands_pcd)
        self.viz.update_geometry(updated_hand_lineset)
        self.viz.poll_events()
        self.viz.update_renderer()



def camera_lineset(center, w, h, color=(0, 0, 0)):
    camera_plane = np.array([center] * 4)
    scaling = np.array([
        [-1*h/2, -1*w/2, 0],
        [-1*h/2, +1*w/2, 0],
        [+1*h/2, +1*w/2, 0],
        [+1*h/2, -1*w/2, 0],
    ])
    camera_plane += scaling

    tunnel_plane = np.array([center] * 4) + scaling * 0.5
    tunnel_plane[:, 2] = -1.5 - center[2]

    tunnel_plane_2 = tunnel_plane.copy()
    tunnel_plane_2[:, 2] = -2 - center[2]

    camera_points = np.vstack([camera_plane, tunnel_plane, tunnel_plane_2])
    camera_lines = list()
    for i in range(3):
        for j in range(4):
            if j == 3:
                camera_lines.append((j + 4*i, i*4))
            else:
                camera_lines.append((j + 4*i, j + 4*i + 1))

            if i > 0:
                camera_lines.append((j+(i-1)*4, j+ i*4))
    colors = o3d.utility.Vector3dVector([color] * len(camera_lines))
    camera_lines = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(camera_points), lines=o3d.utility.Vector2iVector(camera_lines))
    camera_lines.colors = colors
    return camera_lines


class HandLineset():
    def __init__(self, landmarks=None) -> None:
        if landmarks is None:
            landmarks = np.random.rand(21, 3)
        pcd, lineset = hand_lineset(landmarks)
        self.hand_pcd = pcd
        self.hand_lineset = lineset
    
    def update(self, lndmk):
        # convert landmarks to vectors
        landmarks = o3d.utility.Vector3dVector(lndmk)
        self.hand_pcd.points = landmarks
        self.hand_lineset.points = landmarks

        self.hand_lineset.rotate(self.hand_lineset.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
        self.hand_pcd.rotate(self.hand_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
        return self.hand_pcd, self.hand_lineset


def hand_lineset(landmarks):
    # created points
    colors = color_bar(len(landmarks))
    num_landmarks = len(landmarks)
    landmarks = o3d.utility.Vector3dVector(landmarks)

    landmark_pcd = o3d.geometry.PointCloud(landmarks)
    landmark_pcd.colors = colors

    # create lines
    lines = list()
    for connection in Connections.HAND_CONNECTIONS:
        lines.append((connection.start, connection.end))
    lines = o3d.utility.Vector2iVector(lines)

    # return checkboard lineset
    landmark_lineset = o3d.geometry.LineSet(points=landmarks, lines=lines)
    landmark_lineset.colors = color_bar(len(lines))
    return [landmark_lineset, landmark_pcd]

def color_bar(length):
    colormap = getattr(linear, 'viridis').scale(0, 1)
    lower_color = colormap(0)
    upper_color = colormap(1)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    lower_color_rgb = hex_to_rgb(lower_color)
    upper_color_rgb = hex_to_rgb(upper_color)
    color_variations = np.linspace(lower_color_rgb, upper_color_rgb, length) / 255
    return o3d.utility.Vector3dVector(color_variations)
 

class StreamingVisualizer(Visualizer, SyncRedisConsumer):
    def __init__(self, cams=2, R=[], T=[]):
        Visualizer.__init__(self, cams, R, T)
        SyncRedisConsumer.__init__(self, "channel_points_3d")
        

    def consume(self) -> bool:
        count = list()
        self.display_scene()
        for message in self.pubsub.listen():
            messages = self.convert_messages([message])
            message = messages[0]
            # if a terminate message comes thru return false
            # gracefully close window and unsub
            if message.terminate:
                break

            # update landmarks and keep rendering
            self.update_scene(message.points)
            count.append(measure_hand_connections(np.array(message.points)))
            if len(count) > 100:
                np.save("hand_measurements", count)
            print("updates scene")
        self.viz.run()
        


def measure_hand_connections(pcd: np.ndarray):
    print(pcd.shape)
    import mediapipe as mp
    Connections = mp.tasks.vision.HandLandmarksConnections
    
    middle_start_index = Connections.HAND_MIDDLE_FINGER_CONNECTIONS[0].start
    middle_end_index = Connections.HAND_MIDDLE_FINGER_CONNECTIONS[-1].end
    
    index_start_index = Connections.HAND_INDEX_FINGER_CONNECTIONS[0].start
    index_end_index = Connections.HAND_INDEX_FINGER_CONNECTIONS[-1].end

    ring_start_index = Connections.HAND_RING_FINGER_CONNECTIONS[0].start
    ring_end_index = Connections.HAND_RING_FINGER_CONNECTIONS[-1].end

    pinky_start_index = Connections.HAND_PINKY_FINGER_CONNECTIONS[0].start
    pinky_end_index = Connections.HAND_PINKY_FINGER_CONNECTIONS[-1].end

    thumb_start_index = Connections.HAND_THUMB_CONNECTIONS[0].start
    thumb_end_index = Connections.HAND_THUMB_CONNECTIONS[-1].end
    res = [
        1.5 * np.linalg.norm(pcd[thumb_end_index] - pcd[thumb_start_index]),
        1.5 * np.linalg.norm(pcd[index_start_index] - pcd[index_end_index]),
        1.5 * np.linalg.norm(pcd[middle_end_index] - pcd[middle_start_index]), 
        1.5 * np.linalg.norm(pcd[ring_end_index] - pcd[ring_start_index]),
        1.5 * np.linalg.norm(pcd[pinky_end_index] - pcd[pinky_start_index]),
    ]
    print(50 * "*", res)
    return res