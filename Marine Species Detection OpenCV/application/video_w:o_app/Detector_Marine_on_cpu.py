import cv2
import numpy as np
import time
from ultralytics import YOLO  

np.random.seed(16)

class Detector:
    def __init__(self, videoPath, modelPath1, modelPath2):
        self.videoPath = videoPath
        self.modelPath1 = modelPath1
        self.modelPath2 = modelPath2

        # Load YOLOv8 models
        self.model1 = YOLO(self.modelPath1)
        self.model2 = YOLO(self.modelPath2)

        # Initialize color list for bounding box visualization
        self.colorList = np.random.uniform(low=0, high=255, size=(300, 3))  

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

            # Process frame through first YOLOv8 model
            results1 = self.model1(frame, agnostic_nms=True)[0]

            # Process frame through second YOLOv8 model
            results2 = self.model2(frame, agnostic_nms=True)[0]

            combined_results = []

            for result1, result2 in zip(results1, results2):
                # Choose the detection with the highest confidence
                if result1.boxes.conf[0] > result2.boxes.conf[0]:
                    combined_results.append(result1)
                else:
                    combined_results.append(result2)

            if not combined_results or len(combined_results) == 0:
                continue

            for result in combined_results:
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

            # Calculate and display FPS
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime
            frame_count += 1
            total_fps += fps

            # Display average FPS
            avg_fps = total_fps / frame_count
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("Result", frame)

            k = cv2.waitKey(1)
            if k == 27:  # close on ESC key
                cv2.destroyAllWindows()
                break
"""
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

            k = cv2.waitKey(1)
            if k == 27:  # close on ESC key
                cv2.destroyAllWindows()
                break
"""