#!/usr/bin/env python3
"""
webcam_gesture_node.py
Detects real-time hand gestures using MediaPipe Hands from a webcam:
- Extended hand (open palm): STOP
- Folded hand (fist): FOLLOW
Publishes debounced state to ROS 2 topic: /rover/gesture_state (std_msgs/String)
"""

import sys
import math
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Try importing mediapipe
try:
    import mediapipe as mp
except ImportError as e:
    print(f"[webcam_gesture_node] Error importing MediaPipe: {e}")
    sys.exit(1)


class WebcamGestureNode(Node):
    def __init__(self):
        super().__init__('webcam_gesture_node')

        # Declare parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('debounce_frames', 6)
        self.declare_parameter('show_preview', True)
        self.declare_parameter('topic_name', '/rover/gesture_state')

        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.debounce_frames = self.get_parameter('debounce_frames').get_parameter_value().integer_value
        self.show_preview = self.get_parameter('show_preview').get_parameter_value().bool_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        # ROS 2 Publisher
        self.publisher = self.create_publisher(String, topic_name, 10)

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # State tracking
        self.current_published_state = "STOP"  # Default safe state
        self.candidate_state = "STOP"
        self.candidate_counter = 0

        # Video capture
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().warn(f"Could not open camera index {camera_index}. Trying index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            self.get_logger().error("No accessible webcam found. Node will run in standby.")
            self.has_cam = False
        else:
            self.has_cam = True
            self.get_logger().info(f"Webcam opened successfully. Publishing debounced gestures on {topic_name}")

        # Timer loop at ~30 Hz (33ms)
        self.timer = self.create_timer(0.033, self.process_frame)

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def classify_hand(self, landmarks):
        """
        Classifies hand as 'STOP' (most extended), 'FOLLOW' (most folded), or None (transition).
        Uses geometric distance from wrist (landmark 0) to finger tips relative to PIP joints.
        """
        lm = landmarks.landmark
        wrist = lm[0]

        # Finger landmark pairs: (TIP, PIP, MCP)
        fingers = [
            (8, 6, 5),    # Index
            (12, 10, 9),  # Middle
            (16, 14, 13), # Ring
            (20, 18, 17)  # Pinky
        ]

        extended_count = 0

        # Check 4 main fingers
        for tip, pip, mcp in fingers:
            dist_tip = self._dist(wrist, lm[tip])
            dist_pip = self._dist(wrist, lm[pip])
            # Finger extended if tip is further from wrist than PIP joint
            if dist_tip > dist_pip * 1.15 and lm[tip].y < lm[pip].y:
                extended_count += 1

        # Check thumb (tip 4 vs IP 3 and MCP 2)
        dist_thumb_tip = self._dist(wrist, lm[4])
        dist_thumb_ip = self._dist(wrist, lm[3])
        dist_thumb_to_pinky = self._dist(lm[4], lm[17])
        dist_ip_to_pinky = self._dist(lm[3], lm[17])
        if dist_thumb_tip > dist_thumb_ip * 1.1 and dist_thumb_to_pinky > dist_ip_to_pinky:
            extended_count += 1

        # Follow user logic:
        # if most_fingers_are_extended -> STOP
        # elif most_fingers_are_folded -> FOLLOW
        if extended_count >= 4:
            return "STOP", extended_count
        elif extended_count <= 1:
            return "FOLLOW", extended_count
        else:
            return None, extended_count

    def process_frame(self):
        if not self.has_cam:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)
        detected_gesture = None
        ext_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                detected_gesture, ext_count = self.classify_hand(hand_landmarks)
                break  # Process primary hand

        # Temporal debouncing to prevent flickering
        if detected_gesture is not None:
            if detected_gesture == self.candidate_state:
                self.candidate_counter += 1
                if self.candidate_counter >= self.debounce_frames:
                    if self.current_published_state != self.candidate_state:
                        self.current_published_state = self.candidate_state
                        self.get_logger().info(f"[GESTURE COMMITTED] -> {self.current_published_state}")
            else:
                self.candidate_state = detected_gesture
                self.candidate_counter = 1
        else:
            # Ambiguous/in-between frame, reset counter
            self.candidate_counter = max(0, self.candidate_counter - 1)

        # Always publish the current committed state
        msg = String()
        msg.data = self.current_published_state
        self.publisher.publish(msg)

        # GUI Preview
        if self.show_preview:
            # Status badge
            color = (0, 0, 255) if self.current_published_state == "STOP" else (0, 255, 0)
            cv2.rectangle(frame, (10, 10), (320, 90), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, 10), (320, 90), color, 2)
            cv2.putText(frame, f"STATE: {self.current_published_state}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            cv2.putText(frame, f"Fingers Extended: {ext_count}/5", (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Rover Gesture Control (MediaPipe)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Quit requested via preview window.")

    def destroy_node(self):
        if self.has_cam and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamGestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
