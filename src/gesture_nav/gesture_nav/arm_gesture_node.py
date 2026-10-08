#!/usr/bin/env python3
# Arm Gesture Rover Node
#
# Standalone ROS 2 node using ONLY yolo26n-pose.pt (Ultralytics YOLO Pose)
# for both human detection/tracking and arm gesture recognition:
#   - Hands Up      -> STOP   (stands still, unlocks the person)
#   - Arm T-Shape   -> FOLLOW (drives toward person, maintains target_distance)
#   - Arm Cross     -> LOCK   (locks the person crossing arms by YOLO track ID)
#
# ZED 2i camera integration:
#   - Grabs left image + depth with pyzed (no ZED ROS wrapper needed)
#   - VIO positional tracking: publishes /odom (nav_msgs/Odometry) and TF odom -> base_link
#   - Target steering aligned to 3D body center with baseline parallax correction
#
# Press 'q' in the preview window or Ctrl+C in terminal to stop.

import math
import os
import signal
import time
import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


# COCO 17 Pose Keypoint Indices
KPT_NOSE = 0
KPT_LEFT_EYE = 1
KPT_RIGHT_EYE = 2
KPT_LEFT_EAR = 3
KPT_RIGHT_EAR = 4
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_HIP = 11
KPT_RIGHT_HIP = 12
KPT_LEFT_KNEE = 13
KPT_RIGHT_KNEE = 14
KPT_LEFT_ANKLE = 15
KPT_RIGHT_ANKLE = 16


def segments_intersect(p1, p2, p3, p4):
    """Returns True if line segment p1-p2 intersects line segment p3-p4 in 2D."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def classify_arm_gesture(kpts, conf):
    """
    Classifies arm gesture using YOLO-pose 17 keypoints.
    Returns: 'STOP', 'FOLLOW', 'LOCK', or None.
    
    kpts: np.ndarray shape (17, 2), pixel coordinates (x, y)
    conf: np.ndarray shape (17,), confidence scores (0.0 to 1.0)
    """
    if kpts is None or conf is None or len(kpts) < 17 or len(conf) < 17:
        return None

    # Both shoulders must be visible with sufficient confidence
    if conf[KPT_LEFT_SHOULDER] < 0.40 or conf[KPT_RIGHT_SHOULDER] < 0.40:
        return None

    ls = kpts[KPT_LEFT_SHOULDER]
    rs = kpts[KPT_RIGHT_SHOULDER]
    shoulder_dist = float(np.linalg.norm(ls - rs))
    if shoulder_dist < 15.0:
        return None  # Person too far or shoulders collapsed

    # Estimate body scale: use torso height if hips visible, otherwise shoulder distance
    if conf[KPT_LEFT_HIP] > 0.35 and conf[KPT_RIGHT_HIP] > 0.35:
        lh = kpts[KPT_LEFT_HIP]
        rh = kpts[KPT_RIGHT_HIP]
        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0
        torso_height = float(np.linalg.norm(mid_shoulder - mid_hip))
        ref_scale = max(shoulder_dist, torso_height * 0.75)
    else:
        ref_scale = shoulder_dist

    mid_shoulder = (ls + rs) / 2.0
    mid_sh_x, mid_sh_y = mid_shoulder[0], mid_shoulder[1]

    le = kpts[KPT_LEFT_ELBOW]
    re = kpts[KPT_RIGHT_ELBOW]
    lw = kpts[KPT_LEFT_WRIST]
    rw = kpts[KPT_RIGHT_WRIST]

    le_conf = conf[KPT_LEFT_ELBOW]
    re_conf = conf[KPT_RIGHT_ELBOW]
    lw_conf = conf[KPT_LEFT_WRIST]
    rw_conf = conf[KPT_RIGHT_WRIST]

    # -------------------------------------------------------------
    # 1. Hands Up -> STOP
    # Both wrists raised high above shoulders and mid-shoulder line
    # -------------------------------------------------------------
    if lw_conf > 0.35 and rw_conf > 0.35:
        wrists_above_shoulders = (lw[1] < ls[1] - 0.15 * ref_scale) and (rw[1] < rs[1] - 0.15 * ref_scale)
        wrists_above_mid = (lw[1] < mid_sh_y - 0.20 * ref_scale) and (rw[1] < mid_sh_y - 0.20 * ref_scale)

        elbows_elevated = True
        if le_conf > 0.30:
            elbows_elevated = elbows_elevated and (le[1] < mid_sh_y + 0.35 * ref_scale) and (lw[1] < le[1] + 0.15 * ref_scale)
        if re_conf > 0.30:
            elbows_elevated = elbows_elevated and (re[1] < mid_sh_y + 0.35 * ref_scale) and (rw[1] < re[1] + 0.15 * ref_scale)

        if wrists_above_shoulders and wrists_above_mid and elbows_elevated:
            return 'STOP'

    # -------------------------------------------------------------
    # 2. Arm Cross -> LOCK
    # Crossed wrists/arms in front of torso (X-pose or folded arms)
    # -------------------------------------------------------------
    if lw_conf > 0.35 and rw_conf > 0.35 and le_conf > 0.35 and re_conf > 0.35:
        torso_top = mid_sh_y - 0.20 * ref_scale
        torso_bottom = mid_sh_y + 1.40 * ref_scale
        if conf[KPT_LEFT_HIP] > 0.35 and conf[KPT_RIGHT_HIP] > 0.35:
            torso_bottom = max(kpts[KPT_LEFT_HIP][1], kpts[KPT_RIGHT_HIP][1]) + 0.15 * ref_scale

        wrists_in_torso = (torso_top <= lw[1] <= torso_bottom) and (torso_top <= rw[1] <= torso_bottom)
        elbow_span = float(np.linalg.norm(le - re)) > 0.70 * ref_scale

        if wrists_in_torso and elbow_span:
            # Check 1: Forearm segments or arm segments intersect
            seg_cross = segments_intersect(le, lw, re, rw) or segments_intersect(ls, lw, rs, rw)

            # Check 2: Wrists have crossed past each other horizontally
            wrist_x_crossed = (ls[0] < rs[0] and lw[0] > rw[0] - 0.15 * ref_scale) or \
                              (rs[0] < ls[0] and rw[0] > lw[0] - 0.15 * ref_scale)

            # Check 3: Wrists held very close in front of chest center
            wrist_dist = float(np.linalg.norm(lw - rw))
            wrists_close = (wrist_dist < 0.45 * ref_scale) and (abs((lw[0] + rw[0]) / 2.0 - mid_sh_x) < 0.50 * ref_scale)

            # Check 4: Folded arms (each wrist near opposite elbow)
            folded = (np.linalg.norm(lw - re) < 0.65 * ref_scale) and (np.linalg.norm(rw - le) < 0.65 * ref_scale)

            if seg_cross or wrist_x_crossed or wrists_close or folded:
                return 'LOCK'

    # -------------------------------------------------------------
    # 3. Arm T-Shape -> FOLLOW
    # Arms extended horizontally out to both sides
    # -------------------------------------------------------------
    if le_conf > 0.35 and re_conf > 0.35 and lw_conf > 0.35 and rw_conf > 0.35:
        l_horiz = (abs(lw[1] - ls[1]) < 0.35 * ref_scale) and (abs(le[1] - ls[1]) < 0.30 * ref_scale)
        r_horiz = (abs(rw[1] - rs[1]) < 0.35 * ref_scale) and (abs(re[1] - rs[1]) < 0.30 * ref_scale)
        l_ext = abs(lw[0] - ls[0]) > 0.65 * ref_scale
        r_ext = abs(rw[0] - rs[0]) > 0.65 * ref_scale
        span = float(np.linalg.norm(lw - rw)) > 1.80 * ref_scale
        opp_sides = (min(lw[0], rw[0]) < mid_sh_x - 0.40 * ref_scale) and (max(lw[0], rw[0]) > mid_sh_x + 0.40 * ref_scale)

        if l_horiz and r_horiz and l_ext and r_ext and span and opp_sides:
            return 'FOLLOW'

    # -------------------------------------------------------------
    # 4. Right Hand Up -> BACKWARD
    # Person's right wrist (index 10) raised above right shoulder,
    # while left wrist (index 9) is NOT raised.
    # -------------------------------------------------------------
    if rw_conf > 0.35:
        rw_above_shoulder = (rw[1] < rs[1] - 0.15 * ref_scale)
        rw_above_mid = (rw[1] < mid_sh_y - 0.20 * ref_scale)

        # Left arm must NOT be raised (distinguishes from both hands up / STOP)
        lw_not_raised = True
        if lw_conf > 0.35:
            lw_not_raised = (lw[1] >= ls[1] - 0.05 * ref_scale) and (lw[1] >= mid_sh_y - 0.10 * ref_scale)

        re_ok = True
        if re_conf > 0.30:
            re_ok = (rw[1] < re[1] + 0.15 * ref_scale) and (re[1] < mid_sh_y + 0.35 * ref_scale)

        if rw_above_shoulder and rw_above_mid and lw_not_raised and re_ok:
            return 'BACKWARD'

    return None


class ArmGestureNode(Node):
    def __init__(self):
        super().__init__('arm_gesture_node')

        # Resolve default model path: check installed package, then local directory
        try:
            default_model = os.path.join(get_package_share_directory('gesture_nav'), 'yolo26n-pose.pt')
            if not os.path.exists(default_model):
                raise FileNotFoundError
        except Exception:
            default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolo26n-pose.pt'))

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('target_distance', 1.0)  # meters
        self.declare_parameter('show_preview', True)
        self.declare_parameter('camera_x', 0.0)         # meters from base_link
        self.declare_parameter('camera_z', 0.0)         # meters from base_link
        self.declare_parameter('frames_needed', 6)      # consecutive frames to debounce gesture
        self.declare_parameter('device', 'cpu')         # 'cpu' or '0' / 'cuda:0'
        self.declare_parameter('human_height', 1.75)    # meters: average human height for YOLO distance
        self.declare_parameter('focal_length', 0.0)     # pixels: 0.0 means auto-use ZED calibrated fy

        model_path = self.get_parameter('model_path').value
        self.target_distance = float(self.get_parameter('target_distance').value)
        self.show_preview = bool(self.get_parameter('show_preview').value)
        self.camera_x = float(self.get_parameter('camera_x').value)
        self.camera_z = float(self.get_parameter('camera_z').value)
        self.frames_needed = int(self.get_parameter('frames_needed').value)
        self.device = str(self.get_parameter('device').value).lower()
        self.human_height = float(self.get_parameter('human_height').value)
        self.focal_length_param = float(self.get_parameter('focal_length').value)
        # Half precision (FP16) is only supported on GPU / CUDA
        self.half = False if self.device == 'cpu' else True

        # Rover motion settings
        self.deadband = 0.25       # meters deadband around target distance
        self.center_band = 0.05    # normalized fraction of half-width for steering deadband
        self.speed = 30.0          # mt11 rover_run scale (-100..100)
        self.turn_left = -self.speed
        self.turn_right = self.speed

        # ---- ZED 2i camera setup ----
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # needed internally for ZED VIO tracking
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Could not open the ZED camera: ' + repr(err))

        self.image = sl.Mat()

        # ---- ZED visual-inertial odometry ----
        tracking = sl.PositionalTrackingParameters()
        tracking.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3
        tracking.enable_imu_fusion = True
        start = sl.Transform()
        offset = sl.Translation()
        offset.init_vector(self.camera_x, 0.0, self.camera_z)
        start.set_translation(offset)
        tracking.set_initial_world_transform(start)
        err = self.zed.enable_positional_tracking(tracking)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Could not start positional tracking: ' + repr(err))

        self.pose = sl.Pose()
        self.tracking_state = 'OFF'
        self.last_x = None
        self.last_y = None
        self.last_yaw = None
        self.last_time = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # Left camera calibration for baseline parallax and YOLO pinhole distance
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.fx = calib.left_cam.fx
        self.fy = calib.left_cam.fy
        self.cx = calib.left_cam.cx
        self.baseline = calib.get_camera_baseline()
        self.middle_x = self.cx
        res = self.zed.get_camera_information().camera_configuration.resolution
        self.img_w = res.width
        self.img_h = res.height

        # Pinhole focal length: use parameter if given, otherwise auto-use camera calibrated fy
        self.focal_length = self.focal_length_param if self.focal_length_param > 0.0 else (
            self.fy if self.fy > 0.0 else self.img_h * 0.9
        )

        # ---- YOLO26n-Pose setup (No MediaPipe) ----
        self.get_logger().info('Loading YOLO26n-Pose model: %s on device: %s (half=%s)' % (model_path, self.device, self.half))
        self.model = YOLO(model_path)

        # State management
        self.state = 'STOP'           # 'STOP' or 'FOLLOW'
        self.locked_id = None         # Tracking ID of locked person, or None
        self.last_gesture = None
        self.last_candidate_id = None
        self.same_count = 0
        self.frame_count = 0

        # List of detected persons: [(track_id, box, body_center, kpts, confs, gesture)]
        self.people = []

        # ROS publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            'ArmGestureNode initialized. Target dist: %.2f m | YOLO pinhole distance (f=%.1f px, H=%.2f m) on %s. '
            'Gestures: Arms Crossed -> LOCK, T-Shape -> FOLLOW, Right Hand Up -> BACKWARD, Hands Up -> STOP'
            % (self.target_distance, self.focal_length, self.human_height, self.device)
        )

    # ------------------------------------------------------------------
    # YOLO-Pose Person & Gesture Detection
    # ------------------------------------------------------------------
    def find_people_and_gestures(self, frame):
        """
        Runs YOLO pose tracking on the frame.
        Finds persons, extracts 17 body keypoints, computes body center,
        and classifies arm gestures per person.
        """
        track_args = {'device': self.device}
        if self.device != 'cpu':
            track_args['half'] = True

        results = self.model.track(
            frame, classes=[0], conf=0.45, persist=True, verbose=False, **track_args
        )

        boxes = results[0].boxes
        keypoints = results[0].keypoints
        self.people = []

        if boxes.id is None:
            return

        ids = boxes.id.int().tolist()
        corners = boxes.xyxy.tolist()

        kpts_xy = keypoints.xy.cpu().numpy() if keypoints is not None and keypoints.xy is not None else None
        kpts_conf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None

        for i in range(len(ids)):
            box = corners[i]
            person_kpts = kpts_xy[i] if kpts_xy is not None and i < len(kpts_xy) else None
            person_conf = kpts_conf[i] if kpts_conf is not None and i < len(kpts_conf) else None

            body = self.compute_body_center(box, person_kpts, person_conf)
            gesture = classify_arm_gesture(person_kpts, person_conf)

            # Store: (track_id, box, body_center, kpts, confs, gesture)
            self.people.append((ids[i], box, body, person_kpts, person_conf, gesture))

    def compute_body_center(self, box, kpts, confs):
        """
        Computes the center of the torso (shoulders + hips) so that
        arm movements do not pull the rover's tracking center sideways.
        """
        xs = []
        ys = []
        if kpts is not None and confs is not None:
            for k in [KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER, KPT_LEFT_HIP, KPT_RIGHT_HIP]:
                if confs[k] > 0.40:
                    xs.append(kpts[k][0])
                    ys.append(kpts[k][1])

        if len(xs) > 0:
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        # Fallback to bounding box center
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.40)

    # ------------------------------------------------------------------
    # Target Selection & State Transitions
    # ------------------------------------------------------------------
    def person_in_middle(self):
        """Returns the person closest to the image center, or None."""
        best = None
        for person in self.people:
            body_x = person[2][0]
            if best is None or abs(body_x - self.img_w / 2.0) < abs(best[2][0] - self.img_w / 2.0):
                best = person
        return best

    def choose_target(self):
        """Returns the person to track and follow, or None."""
        if self.locked_id is None:
            return self.person_in_middle()

        for person in self.people:
            if person[0] == self.locked_id:
                return person
        return None

    def evaluate_active_gesture(self):
        """
        Determines the candidate gesture and which person performed it.
        Rules:
          - If someone is LOCKED: their gesture takes priority.
            However, ANY person doing 'STOP' (hands up) acts as a safety stop.
          - If NOBODY is locked:
            Any person doing 'LOCK' (arms crossed) triggers a lock on that person.
            Otherwise, the person closest to the image center controls STOP/FOLLOW.
        """
        if not self.people:
            return None, None

        # Check safety STOP: if ANY visible person puts hands up, prioritize STOP
        for person in self.people:
            if person[5] == 'STOP':
                return 'STOP', person[0]

        if self.locked_id is not None:
            # Locked person has control
            for person in self.people:
                if person[0] == self.locked_id:
                    return person[5], self.locked_id
            return None, None
        else:
            # Check if anyone is signaling LOCK (arms crossed)
            for person in self.people:
                if person[5] == 'LOCK':
                    return 'LOCK', person[0]

            # Check person in middle for FOLLOW
            mid_person = self.person_in_middle()
            if mid_person is not None and mid_person[5] is not None:
                return mid_person[5], mid_person[0]

        return None, None

    def update_state(self, gesture, gesturing_id):
        """
        Debounces gesture over self.frames_needed consecutive frames.
        Transitions state and locks/unlocks person tracking ID.
        """
        if gesture is not None and gesture == self.last_gesture and gesturing_id == self.last_candidate_id:
            self.same_count += 1
        else:
            self.same_count = 1

        self.last_gesture = gesture
        self.last_candidate_id = gesturing_id

        if gesture is None or self.same_count != self.frames_needed:
            return

        if gesture == 'LOCK':
            if gesturing_id is not None:
                self.locked_id = gesturing_id
                print('>>> [LOCK] Locked person #%d (arm cross detected)' % self.locked_id)
        elif gesture == 'FOLLOW':
            if self.state != 'FOLLOW':
                self.state = 'FOLLOW'
                print('>>> [STATE] Switched to FOLLOW (arm T-shape detected)')
        elif gesture == 'STOP':
            if self.state != 'STOP' or self.locked_id is not None:
                self.state = 'STOP'
                print('>>> [STATE] Switched to STOP (hands up detected)')
                if self.locked_id is not None:
                    print('>>> [UNLOCK] Unlocked person #%d' % self.locked_id)
                    self.locked_id = None
        elif gesture == 'BACKWARD':
            if self.state != 'BACKWARD':
                self.state = 'BACKWARD'
                print('>>> [STATE] Switched to BACKWARD (right hand up detected)')

    # ------------------------------------------------------------------
    # YOLO Pinhole Distance & Rover Following
    # ------------------------------------------------------------------
    def estimate_yolo_distance(self, box, kpts=None, confs=None):
        """
        Estimates distance (meters) using pinhole camera geometry from the YOLO detection:
            Z = (focal_length * human_height) / box_height

        If the person is close and their feet/head are cut off by the frame borders,
        pose keypoints (torso height) are used to prevent false distance inflation.
        """
        x1, y1, x2, y2 = box
        box_h = max(1.0, float(y2 - y1))

        # Border cutoff safeguard: if box touches image top/bottom, check torso keypoints
        if kpts is not None and confs is not None:
            touches_edge = (y2 >= self.img_h - 6) or (y1 <= 6)
            if touches_edge and confs[KPT_LEFT_SHOULDER] > 0.40 and confs[KPT_RIGHT_SHOULDER] > 0.40 \
               and confs[KPT_LEFT_HIP] > 0.35 and confs[KPT_RIGHT_HIP] > 0.35:
                mid_sh_y = (kpts[KPT_LEFT_SHOULDER][1] + kpts[KPT_RIGHT_SHOULDER][1]) / 2.0
                mid_hip_y = (kpts[KPT_LEFT_HIP][1] + kpts[KPT_RIGHT_HIP][1]) / 2.0
                torso_h = abs(mid_hip_y - mid_sh_y)
                # Anthropometric torso proportion: shoulder-to-hip is ~29% of stature
                if torso_h > 15.0:
                    est_full_h = torso_h / 0.29
                    box_h = max(box_h, est_full_h)

        distance = (self.focal_length * self.human_height) / box_h
        return float(distance)

    def follow_person(self, frame, target):
        """Calculates forward and angular speeds to follow the target."""
        img_h, img_w = frame.shape[:2]

        if target is None:
            if self.locked_id is not None:
                return 0.0, 0.0, 'locked person #%d not seen, waiting' % self.locked_id
            return 0.0, 0.0, 'no person seen'

        track_id, box, body, kpts, confs, _ = target

        # Estimate distance via YOLO pinhole geometry (no ZED depth used)
        distance = self.estimate_yolo_distance(box, kpts, confs)

        distance_error = distance - self.target_distance

        # Parallax compensation for ZED 2i left lens (6 cm baseline offset)
        self.middle_x = self.cx + self.fx * (self.baseline / 2.0) / distance
        side_error = (self.middle_x - body[0]) / (img_w / 2.0)

        forward = 0.0
        turn = 0.0

        if distance_error > self.deadband:
            forward = self.speed
            if side_error > self.center_band:
                turn = self.turn_left
            elif side_error < -self.center_band:
                turn = self.turn_right
        elif distance_error < -self.deadband:
            forward = -self.speed

        return forward, turn, 'person #%d %.2f m away (YOLO), %+.2f from middle' % (track_id, distance, side_error)

    def drive_backward(self, frame, target):
        """
        Drives the rover backward (-self.speed) while keeping the target person
        centered in the camera frame.
        """
        img_h, img_w = frame.shape[:2]

        if target is None:
            if self.locked_id is not None:
                return 0.0, 0.0, 'BACKWARD: locked person #%d not seen, waiting' % self.locked_id
            return 0.0, 0.0, 'BACKWARD: no person seen, waiting'

        track_id, box, body, kpts, confs, _ = target
        distance = self.estimate_yolo_distance(box, kpts, confs)

        # Safety cutoff: if person is already more than 4.5m away, stop backing up
        if distance > 4.5:
            return 0.0, 0.0, 'BACKWARD: reached safe distance (%.2f m), stopped' % distance

        # Parallax compensation for ZED 2i left lens (6 cm baseline offset)
        self.middle_x = self.cx + self.fx * (self.baseline / 2.0) / distance
        side_error = (self.middle_x - body[0]) / (img_w / 2.0)

        forward = -self.speed
        turn = 0.0

        # Steer to keep the person in the camera center while reversing
        if side_error > self.center_band:
            turn = self.turn_left
        elif side_error < -self.center_band:
            turn = self.turn_right

        return forward, turn, 'BACKWARD: person #%d %.2f m away (YOLO), %+.2f from middle' % (track_id, distance, side_error)

    # ------------------------------------------------------------------
    # Visual-Inertial Odometry
    # ------------------------------------------------------------------
    def publish_odometry(self):
        """Fetches ZED VIO tracking pose and publishes /odom and TF."""
        state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
        self.tracking_state = state.name
        if state != sl.POSITIONAL_TRACKING_STATE.OK:
            return

        camera_position = self.pose.get_translation(sl.Translation()).get()
        quat = self.pose.get_orientation(sl.Orientation()).get()
        rotation = self.pose.get_rotation_matrix(sl.Rotation()).r

        camera_offset = np.array([self.camera_x, 0.0, self.camera_z])
        x, y, z = camera_position - rotation.dot(camera_offset)

        yaw = math.atan2(rotation[1][0], rotation[0][0])
        now = self.get_clock().now()
        forward_speed = 0.0
        turn_speed = 0.0

        if self.last_time is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0:
                dx = x - self.last_x
                dy = y - self.last_y
                forward_speed = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
                dyaw = math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
                turn_speed = dyaw / dt

        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw
        self.last_time = now
        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = yaw

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        odom.pose.pose.orientation.x = float(quat[0])
        odom.pose.pose.orientation.y = float(quat[1])
        odom.pose.pose.orientation.z = float(quat[2])
        odom.pose.pose.orientation.w = float(quat[3])
        odom.twist.twist.linear.x = forward_speed
        odom.twist.twist.angular.z = turn_speed
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = odom.header.stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

    # ------------------------------------------------------------------
    # Visualization & Overlays
    # ------------------------------------------------------------------
    def draw_skeleton_and_people(self, frame, target):
        """Draws person bounding boxes, body centers, arm skeletons, and gestures."""
        for person in self.people:
            track_id, box, body, kpts, confs, gesture = person
            x1, y1, x2, y2 = [int(v) for v in box]

            dist = self.estimate_yolo_distance(box, kpts, confs)
            is_target = (target is not None and track_id == target[0])
            if is_target:
                color = (0, 255, 0)
                tag = 'LOCKED' if self.locked_id is not None else 'target'
                label = '%s #%d (%.2fm)' % (tag, track_id, dist)
            else:
                color = (150, 150, 150)
                label = '#%d (%.2fm)' % (track_id, dist)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(body[0]), int(body[1])), 8, color, -1)
            cv2.putText(frame, label, (x1, max(20, y1 - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw gesture badge above person box if recognized
            if gesture is not None:
                if gesture == 'STOP':
                    badge_color = (0, 0, 255)       # Red
                elif gesture == 'FOLLOW':
                    badge_color = (0, 255, 0)       # Green
                elif gesture == 'BACKWARD':
                    badge_color = (255, 120, 0)     # Blue-Orange
                else:  # LOCK
                    badge_color = (0, 215, 255)     # Gold
                badge_text = '[%s]' % gesture
                cv2.putText(frame, badge_text, (x1, max(42, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

            # Draw arm and upper-body skeleton if keypoints available
            if kpts is not None and confs is not None:
                skel_color = (0, 255, 255) if is_target else (180, 180, 180)
                arm_connections = [
                    (KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER),
                    (KPT_LEFT_SHOULDER, KPT_LEFT_ELBOW),
                    (KPT_LEFT_ELBOW, KPT_LEFT_WRIST),
                    (KPT_RIGHT_SHOULDER, KPT_RIGHT_ELBOW),
                    (KPT_RIGHT_ELBOW, KPT_RIGHT_WRIST),
                ]
                for p_a, p_b in arm_connections:
                    if confs[p_a] > 0.35 and confs[p_b] > 0.35:
                        pt1 = (int(kpts[p_a][0]), int(kpts[p_a][1]))
                        pt2 = (int(kpts[p_b][0]), int(kpts[p_b][1]))
                        cv2.line(frame, pt1, pt2, skel_color, 2)

                for joint in [KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER, KPT_LEFT_ELBOW,
                              KPT_RIGHT_ELBOW, KPT_LEFT_WRIST, KPT_RIGHT_WRIST]:
                    if confs[joint] > 0.35:
                        pt = (int(kpts[joint][0]), int(kpts[joint][1]))
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    def draw_hud(self, frame, active_gesture, gesturing_id, info, command_text):
        """Draws top HUD status overlay, middle alignment line, and odometry."""
        img_h, _ = frame.shape[:2]

        # White vertical line showing straight ahead
        x_mid = int(self.middle_x)
        cv2.line(frame, (x_mid, 0), (x_mid, img_h), (255, 255, 255), 1)

        if self.state == 'STOP':
            state_color = (0, 0, 255)
        elif self.state == 'FOLLOW':
            state_color = (0, 255, 0)
        elif self.state == 'BACKWARD':
            state_color = (255, 120, 0)
        else:
            state_color = (255, 255, 255)
        state_text = 'STATE: ' + self.state
        if self.locked_id is not None:
            state_text += '   LOCKED #%d' % self.locked_id
        cv2.putText(frame, state_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)

        # Gesture and debounce counter
        if active_gesture is not None and gesturing_id is not None:
            gest_text = 'Gesture: %s by #%d (held %d/%d)' % (
                active_gesture, gesturing_id, self.same_count, self.frames_needed
            )
        else:
            gest_text = 'Gesture: None'
        cv2.putText(frame, gest_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, info, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, command_text, (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        odom_text = 'odom x=%.2f y=%.2f yaw=%.0f deg [%s]' % (
            self.odom_x, self.odom_y, math.degrees(self.odom_yaw), self.tracking_state
        )
        cv2.putText(frame, odom_text, (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    def command_words(self, forward, turn):
        """Formats speeds into human-readable description."""
        if forward > 0:
            drive = 'FORWARD'
        elif forward < 0:
            drive = 'BACKWARD'
        else:
            drive = 'STAY'

        if turn == self.turn_left:
            steer = 'TURN LEFT'
        elif turn == self.turn_right:
            steer = 'TURN RIGHT'
        else:
            steer = 'STRAIGHT'

        return drive + ' + ' + steer

    # ------------------------------------------------------------------
    # Main Step Execution
    # ------------------------------------------------------------------
    def step(self):
        """Processes one camera frame."""
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return True

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR)
        self.frame_count += 1

        # 1. Odometry
        self.publish_odometry()

        # 2. YOLO26n-Pose Detection and Gesture Classification
        self.find_people_and_gestures(frame)

        # 3. Gesture Evaluation & State Update
        active_gesture, gesturing_id = self.evaluate_active_gesture()
        self.update_state(active_gesture, gesturing_id)

        # 4. Target Following & Speed Command
        self.middle_x = self.cx
        target = self.choose_target()

        if self.state == 'FOLLOW':
            forward, turn, info = self.follow_person(frame, target)
        elif self.state == 'BACKWARD':
            forward, turn, info = self.drive_backward(frame, target)
        else:
            forward, turn, info = 0.0, 0.0, 'stopped by gesture'

        cmd = Twist()
        cmd.linear.x = forward
        cmd.angular.z = turn
        self.cmd_pub.publish(cmd)

        # 5. Terminal output every 10 frames
        command_text = self.command_words(forward, turn)
        if self.frame_count % 10 == 0:
            lock_text = ('locked #%d' % self.locked_id) if self.locked_id is not None else 'not locked'
            print('[%s, %s] %s -> %s' % (self.state, lock_text, info, command_text))
            print('    cmd_vel Twist: linear(x=%.2f, y=%.2f, z=%.2f) angular(x=%.2f, y=%.2f, z=%.2f)'
                  % (cmd.linear.x, cmd.linear.y, cmd.linear.z,
                     cmd.angular.x, cmd.angular.y, cmd.angular.z))
            print('    odom: x=%.2f m  y=%.2f m  yaw=%.1f deg  [tracking %s]'
                  % (self.odom_x, self.odom_y, math.degrees(self.odom_yaw), self.tracking_state))

        # 6. Preview GUI
        if self.show_preview:
            self.draw_skeleton_and_people(frame, target)
            self.draw_hud(frame, active_gesture, gesturing_id, info, command_text)
            cv2.imshow('Arm Gesture Rover', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        return True

    def stop_rover(self):
        """Sends zero Twist to safely stop rover motion."""
        self.cmd_pub.publish(Twist())

    def close(self):
        """Cleans up hardware and ROS node resources."""
        self.stop_rover()
        self.zed.disable_positional_tracking()
        self.zed.close()
        cv2.destroyAllWindows()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ArmGestureNode()
    try:
        while rclpy.ok():
            if not node.step():
                break
    except KeyboardInterrupt:
        pass

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print('Stopping arm gesture rover')
    node.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
