import cv2
import numpy as np
import time
from ultralytics import YOLO  

np.random(42)
class Detector:
    def __init__(self, videoPath, modelPath):
        self.videoPath = videoPath
        self.modelPath = modelPath

        # Load YOLOv8 modelq
        self.model = YOLO(self.modelPath)

        # Initialize color list for bounding box visualization
        self.colorList = np.random.uniform(low=0, high=255, size=(80, 3))  

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        startTime = time.time()
        frame_count = 0
        total_fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame through YOLOv8 with agnostic non-max suppression
            results = self.model(frame, agnostic_nms=True)[0]

            if not results or len(results) == 0:
                continue

            for result in results:

                detection_count = result.boxes.shape[0]

                for i in range(detection_count):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    confidence = float(result.boxes.conf[i].item())
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()

                    x1 = int(bounding_box[0])
                    y1 = int(bounding_box[1])
                    x2 = int(bounding_box[2])
                    y2 = int(bounding_box[3])

                    width = x2 - x1
                    height = y2 - y1

                    color = [int(c) for c in self.colorList[cls]]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                    # Additional lines for detailed box corners
                    lineWidth = min(int(width * 0.3), int(height * 0.3))
                    cv2.line(frame, (x1, y1), (x1 + lineWidth, y1), color, thickness=5)
                    cv2.line(frame, (x1, y1), (x1, y1 + lineWidth), color, thickness=5)
                    cv2.line(frame, (x2, y1), (x2 - lineWidth, y1), color, thickness=5)
                    cv2.line(frame, (x2, y1), (x2, y1 + lineWidth), color, thickness=5)
                    cv2.line(frame, (x1, y2), (x1 + lineWidth, y2), color, thickness=5)
                    cv2.line(frame, (x1, y2), (x1, y2 - lineWidth), color, thickness=5)
                    cv2.line(frame, (x2, y2), (x2 - lineWidth, y2), color, thickness=5)
                    cv2.line(frame, (x2, y2), (x2, y2 - lineWidth), color, thickness=5)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime
            frame_count += 1
            total_fps += fps

            # Display average FPS
            avg_fps = total_fps / frame_count
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("Result", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()