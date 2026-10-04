#!/usr/bin/env python3
# Rover node
# One node that does everything:
#   1. Grabs the left image from the ZED 2i camera with pyzed (no ZED ROS wrapper needed)
#   2. Looks for a hand with MediaPipe:
#        Open palm -> STOP (and unlock the person)
#        Fist      -> FOLLOW
#        Thumbs up -> LOCK the person who shows the thumbs up (by YOLO tracking ID).
#                     From then on only that person is followed, everyone else is ignored.
#   3. Finds and tracks people with YOLO pose (body keypoints). In FOLLOW mode it
#        - measures the distance to the person's body with the ZED depth
#        - drives toward the person until they are target_distance meters away, then stops
#        - turns so the person's body (not the hand) is in the middle of the image
#      The hand gesture is only used for the STOP / FOLLOW / LOCK commands.
#   4. Sends the speed on /cmd_vel (geometry_msgs/Twist) and prints the full Twist,
#      so you can check the logic
#   5. Runs ZED visual-inertial odometry (VIO: camera + IMU) and publishes
#        /odom (nav_msgs/Odometry) and the TF odom -> base_link
#
# Press q in the camera window (or Ctrl+C in the terminal) to quit.

import math
import os
import signal
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class RoverNode(Node):
    def __init__(self):
        super().__init__('rover_node')

        # Settings (can be changed from the launch file)
        # Pose model: finds people and their body points (shoulders, hips, ...)
        default_model = os.path.join(get_package_share_directory('gesture_nav'), 'yolo26n-pose.pt')
        self.declare_parameter('model_path', default_model)
        # MediaPipe hand model (runs on the GPU)
        default_hand_model = os.path.join(get_package_share_directory('gesture_nav'), 'hand_landmarker.task')
        self.declare_parameter('hand_model_path', default_hand_model)
        self.declare_parameter('target_distance', 1.0)  # meters, the rover stops this far from the person
        self.declare_parameter('show_preview', True)
        # Where the camera is on the rover, measured from base_link (the rover center), in meters.
        # x = forward, z = up. The camera is assumed to be centered and level.
        self.declare_parameter('camera_x', 0.0)
        self.declare_parameter('camera_z', 0.0)
        model_path = self.get_parameter('model_path').value
        hand_model_path = self.get_parameter('hand_model_path').value
        self.target_distance = float(self.get_parameter('target_distance').value)
        self.show_preview = self.get_parameter('show_preview').value
        self.camera_x = float(self.get_parameter('camera_x').value)
        self.camera_z = float(self.get_parameter('camera_z').value)

        # Numbers used by the controller
        self.deadband = 0.25       # meters, do not drive if this close to the target distance
        self.center_band = 0.05    # do not turn if the person is this close to the middle (0 to 1)
        # Speed sent in the Twist. The rover (rover_run) reads it as -100 to 100 (percent).
        self.speed = 30.0  # fixed speed for forward, backward and turning
        # Turn direction on this rover (mt11 rover_run): angular.z NEGATIVE = turn LEFT,
        # POSITIVE = turn RIGHT. This is the opposite of the ROS standard.
        # If the rover turns the wrong way, swap the two signs.
        self.turn_left = -self.speed
        self.turn_right = self.speed

        # ---- ZED camera ----
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # VIO needs depth; this is the fastest depth mode
        init.coordinate_units = sl.UNIT.METER
        # Same axes as ROS: x forward, y left, z up
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Could not open the ZED camera: ' + repr(err))
        self.image = sl.Mat()
        self.depth = sl.Mat()       # distance (meters) for every pixel of the left image
        self.depth_data = None

        # ---- ZED visual-inertial odometry (positional tracking) ----
        tracking = sl.PositionalTrackingParameters()
        tracking.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3  # newest ZED tracking (uses an AI model)
        tracking.enable_imu_fusion = True  # use the ZED 2i IMU together with the images
        # Start the camera at its place on the rover, so base_link starts at (0, 0, 0)
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
        # Last odometry, used to compute speed and to print
        self.last_x = None
        self.last_y = None
        self.last_yaw = None
        self.last_time = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # Left camera calibration, used to find where "straight ahead" is in the left image
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.fx = calib.left_cam.fx               # focal length in pixels
        self.cx = calib.left_cam.cx               # image center in pixels
        self.baseline = calib.get_camera_baseline()  # meters between the two lenses (0.12 m)
        self.middle_x = self.cx  # where "straight ahead" is in the image (changes with distance)
        # Image width in pixels (the middle of the image is img_w / 2)
        self.img_w = self.zed.get_camera_information().camera_configuration.resolution.width

        # ---- MediaPipe hand detector, on the GPU ----
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path,
                                              delegate=mp_tasks.BaseOptions.Delegate.GPU),
            running_mode=vision.RunningMode.VIDEO,  # video: uses the last frame to find the hand faster
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        self.hand_time_ms = 0  # MediaPipe needs a time (ms) that grows every frame

        # ---- YOLO person detector ----
        self.get_logger().info('Loading YOLO model: ' + model_path)
        self.model = YOLO(model_path)

        # Start in STOP to be safe
        self.state = 'STOP'
        # A gesture must be seen this many frames in a row before it counts
        self.frames_needed = 6
        self.fingers = 0
        self.last_gesture = None
        self.same_count = 0
        self.frame_count = 0

        # People seen in the last frame: list of (tracking ID, [x1, y1, x2, y2])
        self.people = []
        # Tracking ID of the locked person, or None when nobody is locked
        self.locked_id = None
        # Where the hand (wrist) is in the image, in pixels, or None when no hand is seen
        self.hand_point = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info('Rover node started. Target distance: %.2f m' % self.target_distance)

    # ------------------------------------------------------------------
    # Gesture
    # ------------------------------------------------------------------
    def count_fingers(self, hand):
        # hand is a list of 21 points on the hand. Point 0 is the wrist.
        points = hand
        count = 0

        # Index, middle, ring, pinky: (tip point, middle joint point)
        # A finger is "up" if its tip is higher in the image than its middle joint.
        # (In images, smaller y means higher.)
        for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if points[tip].y < points[joint].y:
                count += 1

        # Thumb: it is "out" if the tip (4) is further from the pinky base (17)
        # than the thumb joint (3) is.
        tip_to_pinky = abs(points[4].x - points[17].x)
        joint_to_pinky = abs(points[3].x - points[17].x)
        if tip_to_pinky > joint_to_pinky:
            count += 1

        return count

    def is_thumbs_up(self, hand):
        points = hand

        # 1. The thumb points up: tip (4) above its joints (3, 2)
        thumb_up = points[4].y < points[3].y < points[2].y

        # 2. The thumb tip is the highest point of the hand
        thumb_highest = True
        for i in range(5, 21):
            if points[i].y < points[4].y:
                thumb_highest = False

        # 3. The other four fingers are curled: each tip is closer to the wrist (0)
        #    than its middle joint is. (This works even when the hand is sideways.)
        def dist_to_wrist(i):
            return math.hypot(points[i].x - points[0].x, points[i].y - points[0].y)

        fingers_curled = True
        for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if dist_to_wrist(tip) > dist_to_wrist(joint):
                fingers_curled = False

        return thumb_up and thumb_highest and fingers_curled

    def read_gesture(self, frame):
        # Returns 'STOP', 'FOLLOW', 'LOCK' or None, and the number of fingers up.
        # Also remembers where the hand is and draws it on the frame.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB colors
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # The time must be bigger than last time, even if two frames come in the same millisecond
        self.hand_time_ms = max(self.hand_time_ms + 1, int(time.monotonic() * 1000))
        result = self.hands.detect_for_video(image, self.hand_time_ms)
        if not result.hand_landmarks:
            self.hand_point = None
            return None, 0

        hand = result.hand_landmarks[0]  # 21 points, x and y from 0 to 1

        # Draw the hand: change the points from 0..1 to pixels
        img_h, img_w = frame.shape[:2]
        pixels = [(int(p.x * img_w), int(p.y * img_h)) for p in hand]
        for a, b in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], (255, 255, 255), 2)
        for x, y in pixels:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Remember where the wrist is (used to lock the person showing thumbs up)
        self.hand_point = (hand[0].x * img_w, hand[0].y * img_h)

        fingers = self.count_fingers(hand)
        # Check thumbs up first, because a thumbs up also looks like "1 finger up"
        if self.is_thumbs_up(hand):
            return 'LOCK', fingers
        if fingers >= 4:
            return 'STOP', fingers
        if fingers <= 1:
            return 'FOLLOW', fingers
        return None, fingers

    def update_state(self, gesture):
        # A gesture only counts when it is seen several frames in a row
        if gesture is not None and gesture == self.last_gesture:
            self.same_count += 1
        else:
            self.same_count = 1
        self.last_gesture = gesture

        # "== frames_needed" makes it happen only once while the hand is held
        if gesture is None or self.same_count != self.frames_needed:
            return

        if gesture == 'LOCK':
            self.lock_person()
        elif gesture != self.state:
            self.state = gesture
            print('>>> Gesture changed to ' + self.state)
            if self.state == 'STOP' and self.locked_id is not None:
                print('>>> Unlocked person #%d' % self.locked_id)
                self.locked_id = None

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------
    def find_people(self, frame):
        # Find people with YOLO and give each one a tracking ID that stays the same
        # from frame to frame. Class 0 is "person" in YOLO.
        # device=0: run on the GPU, half=True: faster 16-bit math
        results = self.model.track(frame, classes=[0], conf=0.45, persist=True, verbose=False,
                                   device=0, half=True)
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        self.people = []
        if boxes.id is None:
            return  # nobody seen
        ids = boxes.id.int().tolist()
        corners = boxes.xyxy.tolist()  # list of [x1, y1, x2, y2]
        for i in range(len(ids)):
            body = self.body_center(corners[i], keypoints, i)
            # Each person is (tracking ID, box, body center point)
            self.people.append((ids[i], corners[i], body))

    def body_center(self, box, keypoints, i):
        # The middle of the shoulders and hips. A raised arm makes the box wider,
        # but it does not move this point, so the rover aims at the body, not the hand.
        # Keypoint numbers: 5 = left shoulder, 6 = right shoulder, 11 = left hip, 12 = right hip
        xs = []
        ys = []
        if keypoints is not None and keypoints.conf is not None:
            points = keypoints.xy[i].tolist()
            confidences = keypoints.conf[i].tolist()
            for k in [5, 6, 11, 12]:
                if confidences[k] > 0.5:  # only points YOLO is sure about
                    xs.append(points[k][0])
                    ys.append(points[k][1])
        if len(xs) > 0:
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        # No body points seen: use the middle of the box, a bit above the center
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.4)

    def distance_at(self, point):
        # Distance (meters) from the ZED depth image around a point, or None.
        # Uses the median of a small square, so one bad pixel does not matter.
        if self.depth_data is None:
            return None
        x, y = int(point[0]), int(point[1])
        h, w = self.depth_data.shape[:2]
        size = 7  # pixels around the point
        patch = self.depth_data[max(0, y - size):min(h, y + size + 1),
                                max(0, x - size):min(w, x + size + 1)]
        good = patch[np.isfinite(patch) & (patch > 0)]  # drop pixels with no depth
        if good.size == 0:
            return None
        return float(np.median(good))

    def person_in_middle(self):
        # Returns the person whose body is closest to the middle, or None
        best = None
        for person in self.people:
            body_x = person[2][0]
            if best is None or abs(body_x - self.img_w / 2) < abs(best[2][0] - self.img_w / 2):
                best = person
        return best

    def person_with_hand(self):
        # Returns (tracking ID, box) of the person whose box contains the hand, or None.
        # If the hand is inside more than one box, pick the box whose center is
        # closest to the hand (left/right).
        if self.hand_point is None:
            return None
        hand_x, hand_y = self.hand_point
        best = None
        for person in self.people:
            x1, y1, x2, y2 = person[1]
            if x1 <= hand_x <= x2 and y1 <= hand_y <= y2:
                if best is None:
                    best = person
                else:
                    center = (x1 + x2) / 2
                    best_center = (best[1][0] + best[1][2]) / 2
                    if abs(center - hand_x) < abs(best_center - hand_x):
                        best = person
        return best

    def lock_person(self):
        # Lock the person who is showing the thumbs up
        person = self.person_with_hand()
        if person is None:
            print('>>> Thumbs up, but the hand is not inside any person box. Not locked.')
            return
        self.locked_id = person[0]
        print('>>> Locked person #%d (the one showing thumbs up)' % self.locked_id)

    def choose_target(self):
        # Returns the person to follow, or None
        if self.locked_id is None:
            # Nobody locked: follow whoever is closest to the middle
            return self.person_in_middle()
        # Locked: only the person with the locked tracking ID counts, everyone else is ignored
        for person in self.people:
            if person[0] == self.locked_id:
                return person
        return None  # locked person not seen right now

    def draw_people(self, frame, target):
        for person in self.people:
            track_id, box, body = person
            x1, y1, x2, y2 = [int(v) for v in box]
            if target is not None and track_id == target[0]:
                color = (0, 255, 0)      # green: the one we follow
                if self.locked_id is not None:
                    label = 'LOCKED #%d' % track_id
                else:
                    label = 'target #%d' % track_id
            else:
                color = (150, 150, 150)  # gray: ignored
                label = '#%d' % track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(body[0]), int(body[1])), 8, color, -1)  # body center
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ------------------------------------------------------------------
    # Person following
    # ------------------------------------------------------------------
    def follow_person(self, frame, target):
        # Returns (forward speed, turn speed, text to show)
        img_h, img_w = frame.shape[:2]

        if target is None:
            if self.locked_id is not None:
                return 0.0, 0.0, 'locked person #%d not seen, waiting' % self.locked_id
            return 0.0, 0.0, 'no person seen'

        track_id, box, body = target

        # Real distance to the person's body, from the ZED depth
        distance = self.distance_at(body)
        if distance is None:
            return 0.0, 0.0, 'person #%d: no depth, stopped' % track_id

        # Positive = person is too far away
        distance_error = distance - self.target_distance
        # We use the LEFT lens, which is half the baseline (6 cm) left of the camera center.
        # So someone straight in front of the rover looks a bit to the RIGHT in the left image.
        # Move the "middle" to where straight ahead really is:
        #   shift (pixels) = focal length * (baseline / 2) / distance
        self.middle_x = self.cx + self.fx * (self.baseline / 2) / distance
        # Positive = person's body is left of the middle. Range is about -1 to 1.
        side_error = (self.middle_x - body[0]) / (img_w / 2)

        forward = 0.0
        turn = 0.0
        if distance_error > self.deadband:
            # Too far: drive toward the person (30) and turn to keep their body in the middle
            forward = self.speed
            if side_error > self.center_band:
                turn = self.turn_left    # body is on the left: turn left
            elif side_error < -self.center_band:
                turn = self.turn_right   # body is on the right: turn right
        elif distance_error < -self.deadband:
            # Too close: back up (-30), no turning
            forward = -self.speed
        # else: at target_distance (+/- deadband): do nothing, forward = 0.0 and turn = 0.0

        return forward, turn, 'person #%d %.2f m away, %+.2f from middle' % (track_id, distance, side_error)

    # ------------------------------------------------------------------
    # Odometry (VIO)
    # ------------------------------------------------------------------
    def publish_odometry(self):
        # Ask the ZED where the camera is, compared to where it started
        state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
        self.tracking_state = state.name
        if state != sl.POSITIONAL_TRACKING_STATE.OK:
            return  # not tracking yet (or lost), do not publish bad data

        camera_position = self.pose.get_translation(sl.Translation()).get()  # [x, y, z]
        quat = self.pose.get_orientation(sl.Orientation()).get()             # [x, y, z, w]
        rotation = self.pose.get_rotation_matrix(sl.Rotation()).r             # 3x3 matrix

        # The camera is camera_x in front of and camera_z above the rover center,
        # so move back from the camera to the rover center (base_link).
        camera_offset = np.array([self.camera_x, 0.0, self.camera_z])
        x, y, z = camera_position - rotation.dot(camera_offset)

        # Heading of the rover (rotation around z), in radians
        yaw = math.atan2(rotation[1][0], rotation[0][0])

        # Speed = change in position / change in time
        now = self.get_clock().now()
        forward_speed = 0.0
        turn_speed = 0.0
        if self.last_time is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0:
                dx = x - self.last_x
                dy = y - self.last_y
                # Only the part of the movement along the rover heading is "forward"
                forward_speed = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
                # Change in heading, kept between -pi and pi
                dyaw = math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
                turn_speed = dyaw / dt
        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw
        self.last_time = now
        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = yaw

        # /odom message
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

        # TF odom -> base_link (same position and orientation)
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
    # Main loop step
    # ------------------------------------------------------------------
    def step(self):
        # Runs once per camera frame. Returns False when we should quit.
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return True

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR)  # ZED gives BGRA
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        self.depth_data = self.depth.get_data()  # 2D array, meters (NaN/inf where there is no depth)
        self.frame_count += 1

        # 0. Odometry
        self.publish_odometry()

        # 1. People (every frame, so the tracking IDs stay the same and we can lock anytime)
        self.find_people(frame)

        # 2. Gesture
        gesture, self.fingers = self.read_gesture(frame)
        self.update_state(gesture)

        # 3. Speed command
        self.middle_x = self.cx  # reset; follow_person() moves it for the person's distance
        target = self.choose_target()
        self.draw_people(frame, target)
        if self.state == 'FOLLOW':
            forward, turn, info = self.follow_person(frame, target)
        else:
            forward, turn, info = 0.0, 0.0, 'stopped by gesture'

        cmd = Twist()
        cmd.linear.x = forward
        cmd.angular.z = turn
        self.cmd_pub.publish(cmd)

        # 4. Print the command and odometry (every 10 frames so the terminal stays readable)
        command_text = self.command_words(forward, turn)
        if self.frame_count % 10 == 0:
            if self.locked_id is None:
                lock_text = 'not locked'
            else:
                lock_text = 'locked #%d' % self.locked_id
            print('[%s, %s] %s -> %s' % (self.state, lock_text, info, command_text))
            print('    cmd_vel Twist: linear(x=%.2f, y=%.2f, z=%.2f) angular(x=%.2f, y=%.2f, z=%.2f)'
                  % (cmd.linear.x, cmd.linear.y, cmd.linear.z,
                     cmd.angular.x, cmd.angular.y, cmd.angular.z))
            print('    odom: x=%.2f m  y=%.2f m  yaw=%.1f deg  [tracking %s]'
                  % (self.odom_x, self.odom_y, math.degrees(self.odom_yaw), self.tracking_state))

        # 5. Show the image
        if self.show_preview:
            self.draw_text(frame, self.fingers, info, command_text)
            cv2.imshow('Rover', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def command_words(self, forward, turn):
        # Turn the numbers into words, e.g. "FORWARD + TURN LEFT"
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

    def draw_text(self, frame, fingers, info, command_text):
        img_h, img_w = frame.shape[:2]

        # Line where "straight ahead" is: the rover tries to keep the person's body on it
        x = int(self.middle_x)
        cv2.line(frame, (x, 0), (x, img_h), (255, 255, 255), 1)

        if self.state == 'STOP':
            color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)  # green
        state_text = 'STATE: ' + self.state
        if self.locked_id is not None:
            state_text += '   LOCKED #%d' % self.locked_id
        cv2.putText(frame, state_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, 'Fingers up: ' + str(fingers), (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, command_text, (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        odom_text = 'odom x=%.2f y=%.2f yaw=%.0f deg [%s]' % (
            self.odom_x, self.odom_y, math.degrees(self.odom_yaw), self.tracking_state)
        cv2.putText(frame, odom_text, (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    def stop_rover(self):
        # A Twist with all zeros means "stand still"
        self.cmd_pub.publish(Twist())

    def close(self):
        self.stop_rover()
        self.zed.disable_positional_tracking()
        self.zed.close()
        self.hands.close()
        cv2.destroyAllWindows()


def main():
    # We handle Ctrl+C ourselves so we can still send a stop command
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RoverNode()
    try:
        while rclpy.ok():
            if not node.step():
                break
    except KeyboardInterrupt:
        pass
    # Ignore more Ctrl+C presses so the camera can close cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print('Stopping rover')
    node.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
