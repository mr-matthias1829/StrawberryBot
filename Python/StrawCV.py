import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# AI MODEL
# =========================
model = YOLO("yolov8n.pt")


# =========================
# CV PIPELINE (HSV)
# =========================
def cv_pipeline(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([179, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

    return output, mask, count


# =========================
# AI PIPELINE (YOLO)
# =========================
def ai_pipeline(frame):
    results = model(frame)

    output = frame.copy()
    count = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"{conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            count += 1

    return output, count


# =========================
# MAIN (WEBCAM ONLY)
# =========================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam niet gevonden!")
        return

    print("Webcam gestart (druk 'q' om te stoppen)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv_out, mask, cv_count = cv_pipeline(frame)
        ai_out, ai_count = ai_pipeline(frame)

        combined = frame.copy()

        cv2.putText(combined, f"CV: {cv_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(combined, f"AI: {ai_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Combined", combined)
        cv2.imshow("AI", ai_out)
        cv2.imshow("CV", cv_out)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()