import cv2
import numpy as np
import math
import os

# this is a example OpenCV script for reading dice
# mostly to check if depencies and such are working


def main():
    base_dir = os.path.dirname(__file__)  # Python/
    img_path = os.path.join(base_dir, "..", "Assets", "dobbelstenen.png")

    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
    frame = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # STEP 1: Detect dice bodies

    # Blur reduces noise and smooths image before thresholding
    # Threshold creates binary image
    _, th = cv2.threshold(
        cv2.GaussianBlur(gray, (15, 15), 0),
        10, 255, cv2.THRESH_BINARY
    )

    # Find contours (outer shapes) + hierarchy (parent-child relationships)
    contours, hierarchy = cv2.findContours(
        th,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Debug image to visualize pip detection per die
    debug = np.zeros_like(frame)

    values = []

    # Only proceed if hierarchy exists
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for cnr, cnt in enumerate(contours):

            # Skip child contours (we only want top-level dice shapes)
            if hierarchy[cnr][3] != -1:
                continue

            # Filter out small noise contours (too small to be a die)
            if cv2.contourArea(cnt) < 1000:
                continue

            # Get bounding box around die
            x, y, w, h = cv2.boundingRect(cnt)

            # STEP 2: Detect pips inside die

            # Blur cropped die region to reduce noise
            crop_blur = cv2.GaussianBlur(gray[y:y + h, x:x + w], (5, 5), 0)

            # High threshold isolates bright pip regions
            _, pip_th = cv2.threshold(crop_blur, 245, 255, cv2.THRESH_BINARY)

            # Place pip mask back into debug image for visualization
            debug[y:y + h, x:x + w] = cv2.cvtColor(pip_th, cv2.COLOR_GRAY2BGR)

            # Find pip contours inside die region
            pip_contours, _ = cv2.findContours(
                pip_th,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            pip_count = 0
            for p in pip_contours:

                # Measure area and perimeter for circularity calculation
                area = cv2.contourArea(p)
                perimeter = cv2.arcLength(p, True)

                # Circularity formula:
                # 1.0 = perfect circle, lower = more irregular shape
                circularity = (
                    4 * math.pi * area / (perimeter * perimeter)
                    if perimeter > 0 else 0
                )

                # Only accept contours that match expected pip size + shape
                if circularity > 0.4 and area > 10:
                    pip_count += 1

                    # Draw detected pip onto debug image
                    cv2.drawContours(
                        debug,
                        [p + np.array([[[x, y]]])],
                        -1,
                        (0, 255, 255),
                        2
                    )

            # Store pip count for this die
            values.append(pip_count)

            # Draw bounding box around die and Display pip count
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(pip_count),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    # Print results to console
    print("Dice values (sorted):", sorted(values))
    print("Total:", sum(values))
    cv2.imshow("Result", frame)
    cv2.imshow("Threshold", th)
    cv2.imshow("Pip Debug", debug)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        pass

    cv2.destroyAllWindows()


main()
