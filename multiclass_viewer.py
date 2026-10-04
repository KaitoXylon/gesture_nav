#!/usr/bin/env python3
# Multi-class object viewer
# Opens the ZED 2i directly (ZED SDK, no ROS needed) and runs the ZED
# multi-class object detection model. Draws a box for every object with its
# class, sub-class (e.g. CAR, DOG, LAPTOP), track ID and distance.
#
# ZED classes: PERSON, VEHICLE, BAG, ANIMAL, ELECTRONICS, FRUIT_VEGETABLE, SPORT
#
# Keys:  q = quit   p = print every object's 3D position (meters)   s = pause
#
# Run:   python3 multiclass_viewer.py                     (fast model)
#        python3 multiclass_viewer.py --model medium      (fast, medium or accurate)
#        python3 multiclass_viewer.py --classes PERSON VEHICLE
#        python3 multiclass_viewer.py --svo file.svo2

import argparse
import cv2
import numpy as np
import pyzed.sl as sl

MODELS = {
    'fast': sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST,
    'medium': sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM,
    'accurate': sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE,
}

# One color (BGR) per class
CLASS_COLORS = {
    sl.OBJECT_CLASS.PERSON: (0, 255, 0),
    sl.OBJECT_CLASS.VEHICLE: (0, 140, 255),
    sl.OBJECT_CLASS.BAG: (255, 0, 255),
    sl.OBJECT_CLASS.ANIMAL: (255, 160, 0),
    sl.OBJECT_CLASS.ELECTRONICS: (255, 255, 0),
    sl.OBJECT_CLASS.FRUIT_VEGETABLE: (0, 255, 255),
    sl.OBJECT_CLASS.SPORT: (0, 0, 255),
}
ALL_CLASSES = list(CLASS_COLORS.keys())


def draw_object(frame, obj, scale):
    box = obj.bounding_box_2d  # 4 corners: top-left, top-right, bottom-right, bottom-left
    if len(box) < 4:
        return
    x1, y1 = int(box[0][0] * scale[0]), int(box[0][1] * scale[1])
    x2, y2 = int(box[2][0] * scale[0]), int(box[2][1] * scale[1])
    color = CLASS_COLORS.get(obj.label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    distance = np.linalg.norm(obj.position)
    if np.isnan(distance):
        text = '%s #%d' % (obj.sublabel.name, obj.id)
    else:
        text = '%s #%d %.2f m' % (obj.sublabel.name, obj.id, distance)
    text += '  %d%%' % obj.confidence

    # Filled label background so the text is readable
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1, th + 6)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(frame, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def print_objects(objects):
    print('--- %d objects ---' % len(objects.object_list))
    for obj in objects.object_list:
        x, y, z = obj.position
        print('#%-3d %-12s %-16s conf=%3.0f  x=%7.3f y=%7.3f z=%7.3f' %
              (obj.id, obj.label.name, obj.sublabel.name, obj.confidence, x, y, z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', default='', help='play an .svo/.svo2 file instead of the live camera')
    parser.add_argument('--model', default='fast', choices=MODELS.keys(), help='ZED multi-class model')
    parser.add_argument('--classes', nargs='+', default=[], choices=[c.name for c in ALL_CLASSES],
                        help='only detect these classes (default: all)')
    parser.add_argument('--confidence', type=float, default=40, help='minimum confidence 0-100')
    parser.add_argument('--max-range', type=float, default=20.0, help='ignore objects farther than this (m)')
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

    # Object tracking needs positional tracking
    tracking = sl.PositionalTrackingParameters()
    tracking.set_as_static = True  # camera is not moving; remove if mounted on a moving rover
    zed.enable_positional_tracking(tracking)

    od_params = sl.ObjectDetectionParameters()
    od_params.detection_model = MODELS[args.model]
    od_params.enable_tracking = True  # keep the same ID for an object across frames
    od_params.max_range = args.max_range
    print('Loading %s model (the first run can take a few minutes to optimize)...' % od_params.detection_model.name)
    err = zed.enable_object_detection(od_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print('Could not enable object detection: ' + repr(err))
        zed.close()
        return

    runtime = sl.ObjectDetectionRuntimeParameters()
    runtime.detection_confidence_threshold = args.confidence
    runtime.object_class_filter = [sl.OBJECT_CLASS[c] for c in args.classes] or ALL_CLASSES

    cam_res = zed.get_camera_information().camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    scale = (display_res.width / cam_res.width, display_res.height / cam_res.height)

    image = sl.Mat()
    objects = sl.Objects()
    paused = False
    print('Running. Keys: q = quit, p = print 3D positions, s = pause')

    while True:
        if not paused:
            if zed.grab() > sl.ERROR_CODE.SUCCESS:
                if args.svo:
                    break  # end of the recording
                continue
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            zed.retrieve_objects(objects, runtime)

            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            counts = {}
            for obj in objects.object_list:
                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    draw_object(frame, obj, scale)
                    counts[obj.label.name] = counts.get(obj.label.name, 0) + 1

            summary = '  '.join('%s:%d' % kv for kv in sorted(counts.items())) or 'nothing detected'
            cv2.putText(frame, 'MULTI-CLASS  ' + summary, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('ZED 2i - Multi-class detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            print_objects(objects)
        if key == ord('s'):
            paused = not paused

    cv2.destroyAllWindows()
    image.free(sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == '__main__':
    main()
