# gesture_nav (ROS 2 Jazzy + ZED 2i)

Control a rover with your hand in front of a ZED 2i camera.

- **Open palm** → `STOP` (rover stands still) and unlock the person
- **Fist** → `FOLLOW` (rover follows the person in front of it, about 1 m away)
- **Thumbs up** → `LOCK` the person who shows the thumbs up (the person whose box contains the hand), using their YOLO tracking ID. Only that ID is followed; everyone else is ignored (gray box). If the locked person disappears, the rover waits.

A gesture must be held for 6 frames in a row to count.

## Node

One node, `rover_node`, does everything:

1. Grabs the left image of the ZED 2i with `pyzed` (HD720 @ 60 fps, `NEURAL_LIGHT` depth for the odometry). The ZED ROS wrapper is **not** used.
2. Reads the hand gesture with MediaPipe `HandLandmarker` on the **GPU** (open palm = `STOP`, fist = `FOLLOW`, thumbs up = `LOCK`). YOLO also runs on the GPU (`device=0`, half precision).
3. In `FOLLOW` mode it finds a person with YOLO, turns to keep them on the middle line of the image, and drives forward/backward to stay `target_distance` meters away.
4. Publishes the speed on `/cmd_vel` (`geometry_msgs/Twist`: `linear.x` = forward/backward, `angular.z` = turn) and prints it:
   ```
   [FOLLOW, locked #1] person #1 3.00 m away, -0.50 from middle -> FORWARD + TURN RIGHT
       cmd_vel Twist: linear(x=30.00, y=0.00, z=0.00) angular(x=0.00, y=0.00, z=-30.00)
       odom: x=0.00 m  y=0.00 m  yaw=0.0 deg  [tracking OK]
   ```
5. Runs ZED visual-inertial odometry (camera + IMU) and publishes `/odom` (`nav_msgs/Odometry`, frame `odom` -> `base_link`) and the same transform on TF. Odometry is only published while tracking is `OK`.

The camera window shows the hand, the state, the finger count, the person box, the middle line and the command.

## Setup (once)

```bash
sudo apt install python3.12-venv
python3 -m venv --system-site-packages ~/gesture_nav_venv
~/gesture_nav_venv/bin/pip install "numpy<2" "opencv-python<4.12" "opencv-contrib-python<4.12" mediapipe==0.10.21 ultralytics "lap>=0.5.12"
# pyzed: copy it from the ZED SDK install (python3 /usr/local/zed/get_python_api.py)
cp -r ~/.local/lib/python3.12/site-packages/pyzed* ~/gesture_nav_venv/lib/python3.12/site-packages/
# MediaPipe hand model (GPU), installed with the package
curl -Lo src/gesture_nav/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Build

```bash
cd ~/gesture_nav
source /opt/ros/jazzy/setup.bash
colcon build
```

## Run

Make sure nothing else uses the camera (ZED wrapper, ZED Explorer, the viewers below), then:

```bash
cd ~/gesture_nav
./run_all.sh
# or with options:
./run_all.sh target_distance:=2.0 show_preview:=false
# camera position on the rover, measured from the rover center (m): x forward, z up
./run_all.sh camera_x:=0.25 camera_z:=0.40
# or directly:
source env.sh && ros2 run gesture_nav rover_node
```

Press `q` in the window or Ctrl+C to quit (a stop command is sent).

## How the following works

- YOLO pose (`yolo26n-pose.pt`) finds and tracks people (each person keeps a track id, `#1`, `#2`, ...) and their body points. Without a lock, the person whose body is closest to the image center is followed; with a lock, only the locked id.
- The rover aims at the **body center** (middle of shoulders and hips), not the box center, so a raised hand does not make it turn.
- The distance is the **ZED depth** at the body center (real meters). If there is no depth there, the rover stops.
- The rover stops when the person is `target_distance` m away (default 1.0, `./run_all.sh target_distance:=2.0`), with a ±0.25 m deadband so it does not jitter.
- Speed is on the -100..100 scale that `rover_run` (mt11-controls) expects, hard-coded in `rover_node.py` (`self.speed = 30.0`):
  - person too far (more than `target_distance` + 0.25 m): forward 30, and turn ±30 to keep their body in the middle
  - person at `target_distance` ± 0.25 m: **nothing** (0.0, 0.0)
  - person too close: backward -30, no turning
- Turn sign matches the mt11 rover, **not** the ROS standard: `angular.z` negative = turn left, positive = turn right (`self.turn_left` / `self.turn_right` in `rover_node.py`, swap them if your rover turns the wrong way).
- "Straight ahead" is corrected for the left ZED lens being 6 cm left of the camera center (the white line in the window shows it).
- If nobody is seen, the rover stops.

## Body 38 pose viewer (ZED SDK, no ROS)

Shows the 38-keypoint ZED skeleton (`BODY_38`) on the live left image.
Stop `rover_node` (and the ZED ROS wrapper) first, because only one program can open the camera.

```bash
python3 body38_viewer.py              # fast model
python3 body38_viewer.py --accurate   # slower, more accurate
python3 body38_viewer.py --no-labels  # hide the keypoint numbers
```

Keys: `q` quit, `p` print the 3D position (meters, x right / y down / z forward) of every keypoint, `s` pause.
Colors: blue = left side, orange = right side, green = center.

## Multi-class object viewer (ZED SDK, no ROS)

Runs the ZED `MULTI_CLASS_BOX` detector. It draws a box for each object with its sub-class (CAR, DOG, LAPTOP, ...), track ID, distance and confidence.
Classes: PERSON, VEHICLE, BAG, ANIMAL, ELECTRONICS, FRUIT_VEGETABLE, SPORT.

```bash
python3 multiclass_viewer.py                       # fast model, all classes
python3 multiclass_viewer.py --model accurate      # fast | medium | accurate
python3 multiclass_viewer.py --classes PERSON VEHICLE --confidence 50 --max-range 10
```

Keys: `q` quit, `p` print every object's 3D position (meters), `s` pause.
