import time
import logging

from dynamixel import (
    ServoController,
    JointServo,
    WheelServo,
    safe_shutdown,
)

logging.basicConfig(level=logging.INFO)

# DO NOT USE THIS CODE IN PRODUCTION
# IT IS MEANT TO SERVE AS A EXAMPLE ON HOW TO USE THE DYNAMIXEL.PY
# CREDITS FOR GPT FOR GENERATING THIS EXAMPLE




# IMPORTANT:
# IDs must match the IDs configured on the actual servos.
#
# Example bus:
#   ID 1 = shoulder
#   ID 2 = elbow
#   ID 3 = left wheel
#   ID 4 = right wheel


def main():

    # Opens GPIO + serial automatically
    with ServoController() as ctrl:

        # ------------------------------------------------------------------
        # SAFETY GATE
        # ------------------------------------------------------------------
        #
        # Default is 0.0 = NOTHING MOVES.
        # You must explicitly enable movement.
        #
        # 0.25 = 25% speed
        # 1.0  = full speed
        #
        # for production, this is to make sure nothing breaks because min/max might be incorrect for example
        ctrl.set_speed_scale(0.35)

        # ------------------------------------------------------------------
        # CREATE SERVOS
        # ------------------------------------------------------------------
#13 5 3 4 //8
        shoulder = JointServo(
            ctrl,
            servo_id=13,
            min_pos=350,
            max_pos=700,
            speed=120,
            label="shoulder",
        )

        elbow = JointServo(
            ctrl,
            servo_id=5,
            min_pos=250,
            max_pos=800,
            speed=100,
            label="elbow",
        )

        left_wheel = WheelServo(
            ctrl,
            servo_id=3,
            max_speed=400,
            label="left_wheel",
        )

        right_wheel = WheelServo(
            ctrl,
            servo_id=4,
            max_speed=400,
            label="right_wheel",
        )

        try:

            # ==============================================================
            # ARM MOVEMENT
            # ==============================================================

            print("Move arm to center")

            shoulder.move_to_center(wait=True)
            elbow.move_to_center(wait=True)

            time.sleep(1)

            # Move using percentages
            # 0.0 = min limit
            # 1.0 = max limit

            print("Lift arm")

            shoulder.move_to_percent(0.8)
            elbow.move_to_percent(0.2, wait=True)

            time.sleep(1)

            # Absolute raw positions
            print("Move to exact positions")

            shoulder.move_to(500)
            elbow.move_to(650, wait=True)

            # ==============================================================
            # DRIVE MOVEMENT
            # ==============================================================

            print("Drive forward")

            # Depending on wheel orientation,
            # you may need to invert one side.
            left_wheel.spin(250, clockwise=True)
            right_wheel.spin(250, clockwise=False)

            time.sleep(2)

            print("Turn left")

            left_wheel.spin(120, clockwise=True)
            right_wheel.spin(300, clockwise=False)

            time.sleep(1.5)

            print("Stop wheels")

            left_wheel.stop()
            right_wheel.stop()

            # ==============================================================
            # SENSOR / STATUS READS
            # ==============================================================

            shoulder_pos = shoulder.get_position()
            shoulder_load = shoulder.get_load()

            print("Shoulder position:", shoulder_pos)
            print("Shoulder load:", shoulder_load)

            # ==============================================================
            # GLOBAL SPEED SCALING
            # ==============================================================

            print("Slow mode")

            ctrl.set_speed_scale(0.1)

            shoulder.move_to_percent(1.0, wait=True)

            print("Fast mode")

            ctrl.set_speed_scale(1.0)

            shoulder.move_to_percent(0.0, wait=True)

        finally:

            # ALWAYS DO THIS
            #
            # Stops movement
            # disables torque
            # saves state cleanly

            safe_shutdown(
                shoulder,
                elbow,
                left_wheel,
                right_wheel,
            )


if __name__ == "__main__":
    main()