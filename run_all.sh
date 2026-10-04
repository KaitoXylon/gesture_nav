#!/usr/bin/env bash
# Starts the rover node (run "colcon build" first).
# Extra arguments are passed to the launch file, e.g. ./run_all.sh target_distance:=2.0
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
# Use the Python packages (mediapipe, ultralytics, pyzed) from the virtual environment
export PYTHONPATH="$HOME/gesture_nav_venv/lib/python3.12/site-packages:$PYTHONPATH"
# Ignore ~/.local packages (it has numpy 2, which breaks cv_bridge and mediapipe)
export PYTHONNOUSERSITE=1
ros2 launch gesture_nav gesture_nav.launch.py "$@"
