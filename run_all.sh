#!/usr/bin/env bash
# run_all.sh - Launches both webcam gesture detector and YOLO human follower
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source /opt/ros/humble/setup.bash
source /home/masuk/gesture_nav_venv/bin/activate
source "$DIR/install/setup.bash"

echo "=========================================================="
echo " Starting Gesture Navigation & Human Follower System"
echo " - Gesture Topic:   /rover/gesture_state"
echo " - Camera Topic:    /zed2i/zed_node/left/image_rect_color"
echo " - Command Output:  /cmd_vel"
echo "=========================================================="
ros2 launch gesture_nav gesture_nav.launch.py "$@"
