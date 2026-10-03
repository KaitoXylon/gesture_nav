#!/usr/bin/env python3
"""
human_follower_node.py
Continuous YOLO26n human detection and 3-meter distance follower for Drubotara Rover.
- Subscribes to /rover/gesture_state (STOP vs FOLLOW).
- Subscribes to ZED 2i camera topic (/zed2i/zed_node/left/image_rect_color).
- Uses YOLO26n to detect human ('person' class).
- Estimates distance via pinhole projection geometry.
- Publishes /cmd_vel to maintain 3.0m distance when in FOLLOW mode.
"""

import sys
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"[human_follower_node] Error importing ultralytics: {e}")
    sys.exit(1)


class HumanFollowerNode(Node):
    def __init__(self):
        super().__init__('human_follower_node')

        # Declare parameters
        self.declare_parameter('model_path', 'yolo26n.pt')
        self.declare_parameter('camera_topic', '/zed2i/zed_node/left/image_rect_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('gesture_topic', '/rover/gesture_state')
        self.declare_parameter('target_distance_m', 1.0)
        self.declare_parameter('distance_deadband_m', 0.25)
        self.declare_parameter('human_height_m', 1.75)
        self.declare_parameter('focal_length_px', 550.0)
        self.declare_parameter('kp_linear', 0.6)
        self.declare_parameter('kp_angular', 1.2)
        self.declare_parameter('max_linear_speed', 1.2)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('show_preview', True)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        gesture_topic = self.get_parameter('gesture_topic').get_parameter_value().string_value
        self.target_distance = self.get_parameter('target_distance_m').get_parameter_value().double_value
        self.deadband = self.get_parameter('distance_deadband_m').get_parameter_value().double_value
        self.human_height = self.get_parameter('human_height_m').get_parameter_value().double_value
        self.focal_length = self.get_parameter('focal_length_px').get_parameter_value().double_value
        self.kp_linear = self.get_parameter('kp_linear').get_parameter_value().double_value
        self.kp_angular = self.get_parameter('kp_angular').get_parameter_value().double_value
        self.max_linear = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.show_preview = self.get_parameter('show_preview').get_parameter_value().bool_value

        # Initialize CV Bridge
        self.bridge = CvBridge() if CvBridge else None

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.get_logger().info("YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().warn(f"Could not load {model_path} directly ({e}). Falling back to yolov8n.pt...")
            self.model = YOLO('yolov8n.pt')

        # State tracking
        self.current_gesture = "STOP"
        self.latest_cv_image = None
        self.new_image_received = False

        # ROS 2 Subscriptions & Publishers
        self.gesture_sub = self.create_subscription(
            String, gesture_topic, self.gesture_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Main tracking loop at 20 Hz (50ms)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"HumanFollowerNode started. Subscribed to {camera_topic}, publishing on {cmd_vel_topic}")

    def gesture_callback(self, msg: String):
        prev = self.current_gesture
        self.current_gesture = msg.data.strip().upper()
        if self.current_gesture != prev:
            self.get_logger().info(f"[MODE TRANSITION] Gesture: {prev} -> {self.current_gesture}")
            if self.current_gesture == "STOP":
                self.stop_rover()

    def image_callback(self, msg: Image):
        try:
            if self.bridge:
                self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                # Direct numpy decoding fallback
                dtype = np.uint8
                channels = 3 if 'rgb' in msg.encoding.lower() or 'bgr' in msg.encoding.lower() else 1
                img = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, channels))
                if 'rgb' in msg.encoding.lower():
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.latest_cv_image = img
            self.new_image_received = True
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def stop_rover(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_cmd)

    def control_loop(self):
        if self.current_gesture == "STOP":
            self.stop_rover()
            if self.show_preview and self.latest_cv_image is not None:
                self.render_debug_view(self.latest_cv_image, None, 0.0, "STOPPED")
            return

        if self.latest_cv_image is None or not self.new_image_received:
            return

        frame = self.latest_cv_image.copy()
        self.new_image_received = False
        img_h, img_w, _ = frame.shape

        # Run inference (person class is 0 in COCO)
        results = self.model.predict(source=frame, classes=[0], verbose=False, conf=0.45)
        boxes = results[0].boxes if len(results) > 0 else []

        best_target = None
        min_center_dist = float('inf')

        # Select primary human: closest to image center
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2.0
            dist_from_center = abs(cx - (img_w / 2.0))
            if dist_from_center < min_center_dist:
                min_center_dist = dist_from_center
                best_target = xyxy

        if best_target is None:
            # Human lost in FOV: safely halt
            self.stop_rover()
            if self.show_preview:
                self.render_debug_view(frame, None, 0.0, "FOLLOWING (SEARCHING)")
            return

        x1, y1, x2, y2 = best_target
        box_h = max(1.0, y2 - y1)
        box_w = max(1.0, x2 - x1)
        box_cx = (x1 + x2) / 2.0

        # Pinhole distance estimation: Z = (f * H) / h
        # Auto-calibrate focal length if default: f = img_h / (2 * tan(30 deg)) ~ 0.866 * img_h
        focal_length = self.focal_length if self.focal_length > 0 else (img_h * 0.9)
        estimated_distance = (focal_length * self.human_height) / box_h

        # Compute Errors
        # Distance error: positive means human is further than target_distance
        dist_error = estimated_distance - self.target_distance

        # Bearing error: normalized [-1, 1], positive when target is LEFT of frame center
        # (ZED coordinate / standard yaw: positive angular.z turns Left)
        center_x = img_w / 2.0
        bearing_error = (center_x - box_cx) / (img_w / 2.0)

        # Velocity Commands
        cmd = Twist()

        # Distance control with deadband
        if abs(dist_error) > self.deadband:
            cmd.linear.x = float(np.clip(dist_error * self.kp_linear, -0.4, self.max_linear))
        else:
            cmd.linear.x = 0.0

        # Angular steering control
        if abs(bearing_error) > 0.05:
            cmd.angular.z = float(np.clip(bearing_error * self.kp_angular, -self.max_angular, self.max_angular))
        else:
            cmd.angular.z = 0.0

        # Publish command to rover
        self.cmd_vel_pub.publish(cmd)

        # Render debug visualizer
        if self.show_preview:
            self.render_debug_view(frame, best_target, estimated_distance, f"FOLLOWING ({estimated_distance:.2f}m)")

    def render_debug_view(self, frame, target_box, distance, status_text):
        vis = frame.copy()
        h, w, _ = frame.shape

        # Center reticle
        cv2.line(vis, (int(w / 2), 0), (int(w / 2), h), (80, 80, 80), 1)
        cv2.line(vis, (0, int(h / 2)), (w, int(h / 2)), (80, 80, 80), 1)

        # Draw detected target box
        if target_box is not None:
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 6, (0, 255, 255), -1)
            cv2.putText(vis, f"Human: {distance:.2f} m", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Status badge overlay
        badge_color = (0, 0, 255) if "STOPPED" in status_text else (0, 255, 0)
        cv2.rectangle(vis, (10, 10), (360, 80), (30, 30, 30), -1)
        cv2.rectangle(vis, (10, 10), (360, 80), badge_color, 2)
        cv2.putText(vis, f"MODE: {status_text}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, badge_color, 2)
        cv2.putText(vis, f"Target: {self.target_distance:.1f}m | Deadband: {self.deadband:.2f}m", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("ZED 2i YOLO Human Follower", vis)
        cv2.waitKey(1)

    def destroy_node(self):
        self.stop_rover()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
