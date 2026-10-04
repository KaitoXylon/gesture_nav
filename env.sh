# Source this before "ros2 run gesture_nav ...":   source ~/gesture_nav/env.sh
source /opt/ros/jazzy/setup.bash
source "$HOME/gesture_nav/install/setup.bash"
# Python packages (mediapipe, ultralytics, numpy<2) from the virtual environment
export PYTHONPATH="$HOME/gesture_nav_venv/lib/python3.12/site-packages:$PYTHONPATH"
# Ignore ~/.local packages (numpy 2 there breaks cv_bridge and mediapipe)
export PYTHONNOUSERSITE=1
