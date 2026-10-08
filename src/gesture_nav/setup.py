from setuptools import setup

package_name = 'gesture_nav'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'yolo26n.pt', 'yolo26n-pose.pt', 'hand_landmarker.task']),
        ('share/' + package_name + '/launch', [
            'launch/gesture_nav.launch.py',
            'launch/arm_gesture_nav.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='masuk',
    maintainer_email='masuk@todo.todo',
    description='Arm and hand gesture control and YOLO person following with a ZED 2i camera (pyzed)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'rover_node = gesture_nav.rover_node:main',
            'arm_gesture_node = gesture_nav.arm_gesture_node:main',
        ],
    },
)
