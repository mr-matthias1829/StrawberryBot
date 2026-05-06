import cv2
import numpy as np
import os


def main():
    base_dir = os.path.dirname(__file__)
    img_path = os.path.join(base_dir, "..", "Assets", "StrawberryPlant1.jpg")

    frame = cv2.imread(img_path)
    if frame is None:
        print("Image not found")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)

    # This only seems to backfire, hence it's commented out
    #mask = cv2.erode(mask, kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Each strawberry's center is a local peak — even when two touch,
    # each has its own local maximum. This dilation trick finds them all.
    kernel_lm = np.ones((21, 21), np.uint8)  # Controls min distance between peaks
    local_max = cv2.dilate(dist, kernel_lm)
    sure_fg = np.uint8((dist == local_max) & (dist > 8))  # >8 filters out background noise

    # Dilate seeds slightly so watershed has a stable region to grow from
    sure_fg = cv2.dilate(sure_fg, np.ones((50, 50), np.uint8))

    unknown = cv2.subtract(mask, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame, markers)


    # Visualize watershed labels with depth ordering: front=1, back=highest.
    unique_labels = sorted([l for l in np.unique(markers) if l > 1])

    # Find center Y position of each label to determine depth order
    label_to_order = {}
    for i, label in enumerate(unique_labels):
        y_coords = np.where(markers == label)[0]
        if len(y_coords) > 0:
            center_y = np.mean(y_coords)
            label_to_order[label] = (i + 1, center_y)

    # Sort by Y position: lower Y (top/front) gets lower order number
    sorted_by_y = sorted(label_to_order.items(), key=lambda x: x[1][1])
    label_to_order = {label: i + 1 for i, (label, _) in enumerate(sorted_by_y)}

    # Create grayscale visualization and add order numbers
    markers_view = np.zeros_like(markers, dtype=np.uint8)
    for label in unique_labels:
        order_num = label_to_order[label]
        markers_view[markers == label] = int(order_num * 255 / len(unique_labels))

    # Add text labels showing order numbers
    markers_view_display = cv2.cvtColor(markers_view, cv2.COLOR_GRAY2BGR)
    for label, order_num in label_to_order.items():
        y_coords, x_coords = np.where(markers == label)
        if len(x_coords) > 0:
            center_x, center_y = int(np.median(x_coords)), int(np.median(y_coords))
            cv2.putText(markers_view_display, str(order_num), (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)



    output = frame.copy()
    count = 0

    for label in np.unique(markers):
        if label <= 1:
            continue

        obj_mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(
            obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1: # adjust as needed, will likely need to be made dynamic
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

    #cv2.imshow("Original", frame)
    #cv2.imshow("Mask", mask) # markers view is better
    cv2.imshow("Distance", dist_view)
    cv2.imshow("Watershed Labels (Ordered)", markers_view_display)
    cv2.imshow("Result", output) # the only one we truely care about

    print("Detected objects:", count)


    # should be removed at one point when we make this autonomous
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()