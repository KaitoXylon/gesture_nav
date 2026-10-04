#!/usr/bin/env python3
# Body 38 viewer
# Opens the ZED 2i directly (ZED SDK, no ROS needed), finds people with
# ZED Body Tracking in BODY_38 format, and draws the 38-point skeleton
# on the live camera image.
#
# Keys:  q = quit   p = print the 3D points (meters) of every person   s = pause
#
# Run:   python3 body38_viewer.py            (live camera)
#        python3 body38_viewer.py --accurate (slower, better model)
#        python3 body38_viewer.py --svo file.svo2

import argparse
import cv2
import numpy as np
import pyzed.sl as sl

# Colors (BGR) for left side, right side and the middle of the body
LEFT_COLOR = (255, 160, 0)
RIGHT_COLOR = (0, 140, 255)
CENTER_COLOR = (0, 255, 0)


def part_color(name):
    if name.startswith('LEFT'):
        return LEFT_COLOR
    if name.startswith('RIGHT'):
        return RIGHT_COLOR
    return CENTER_COLOR


def is_valid(point):
    # The SDK marks keypoints it could not find with NaN or negative values
    return not np.isnan(point).any() and point[0] >= 0 and point[1] >= 0


def draw_body(frame, body, scale, show_labels):
    points = [(p[0] * scale[0], p[1] * scale[1]) for p in body.keypoint_2d]

    # Bones
    for part_a, part_b in sl.BODY_38_BONES:
        a = points[part_a.value]
        b = points[part_b.value]
        if is_valid(a) and is_valid(b):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     part_color(part_b.name), 2, cv2.LINE_AA)

    # Joints
    for i, p in enumerate(points):
        if not is_valid(p):
            continue
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, part_color(sl.BODY_38_PARTS(i).name), -1)
        if show_labels:
            cv2.putText(frame, str(i), (int(p[0]) + 4, int(p[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Person id and distance above the head
    head = points[sl.BODY_38_PARTS.NOSE.value]
    if is_valid(head):
        distance = np.linalg.norm(body.position)
        cv2.putText(frame, 'ID %d  %.2f m' % (body.id, distance),
                    (int(head[0]) - 40, int(head[1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_points(bodies):
    for body in bodies.body_list:
        print('--- Person ID %d (confidence %.0f) ---' % (body.id, body.confidence))
        for i, p in enumerate(body.keypoint):
            print('%2d %-20s x=%7.3f y=%7.3f z=%7.3f' % (i, sl.BODY_38_PARTS(i).name, p[0], p[1], p[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', default='', help='play an .svo/.svo2 file instead of the live camera')
    parser.add_argument('--accurate', action='store_true', help='use HUMAN_BODY_ACCURATE (slower)')
    parser.add_argument('--no-labels', action='store_true', help='do not draw keypoint numbers')
    args = parser.parse_args()

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    # x right, y down, z forward (same as the image)
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    if args.svo:
        init.set_from_svo_file(args.svo)

    err = zed.open(init)
    if err > sl.ERROR_CODE.SUCCESS:
        print('Could not open the ZED camera: ' + repr(err))
        return

    # Body tracking needs positional tracking
    tracking = sl.PositionalTrackingParameters()
    tracking.set_as_static = True  # camera is not moving; remove if mounted on a moving rover
    zed.enable_positional_tracking(tracking)

    body_params = sl.BodyTrackingParameters()
    body_params.body_format = sl.BODY_FORMAT.BODY_38
    body_params.detection_model = (sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE if args.accurate
                                   else sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST)
    body_params.enable_tracking = True      # keep the same ID for a person across frames
    body_params.enable_body_fitting = True  # smoother skeleton
    print('Loading body tracking model (the first run can take a few minutes to optimize)...')
    err = zed.enable_body_tracking(body_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print('Could not enable body tracking: ' + repr(err))
        zed.close()
        return

    runtime = sl.BodyTrackingRuntimeParameters()
    runtime.detection_confidence_threshold = 40

    cam_res = zed.get_camera_information().camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    scale = (display_res.width / cam_res.width, display_res.height / cam_res.height)

    image = sl.Mat()
    bodies = sl.Bodies()
    paused = False
    print('Running. Keys: q = quit, p = print 3D points, s = pause')

    while True:
        if not paused:
            if zed.grab() > sl.ERROR_CODE.SUCCESS:
                if args.svo:
                    break  # end of the recording
                continue
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            zed.retrieve_bodies(bodies, runtime)

            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            for body in bodies.body_list:
                if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    draw_body(frame, body, scale, not args.no_labels)
            cv2.putText(frame, 'BODY_38  people: %d' % len(bodies.body_list), (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('ZED 2i - Body 38', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            print_points(bodies)
        if key == ord('s'):
            paused = not paused

    cv2.destroyAllWindows()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == '__main__':
    main()
