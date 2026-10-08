# Learning Ledger: gesture_nav (gesture-controlled person-following rover)

## Understanding
- [ ] Goal & motivation
- [ ] System boundary (inputs/outputs/components)
- [ ] Data flow
- [ ] Core concepts / math
- [ ] Mechanism
- [ ] Assumptions
- [ ] Failure modes
- [ ] Integration points
- [ ] Validation strategy
- [ ] Critical code logic

## Misconceptions caught
- 2026-10-04 Thought the 1 m → 2.3 m distance error came from a "loose" YOLO box → it came from the box being cut off at the 720 px image edge (a loose box would make the distance *smaller*, not bigger).
- 2026-10-04 Thought linear.x = 30 is a raw PWM value → it is a percent (−100..100) that mt11 rover_run maps to PWM 1500 ± range. Also: this breaks the ROS SI-unit convention (m/s).
- 2026-10-04 Missed that rover_node also publishes /odom + TF (ZED VIO).

## Questions I learned to ask

## Skipped (debt)
