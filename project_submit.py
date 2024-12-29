from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
import cv2
import os

model = YOLO("best.pt")

class_names = {
    0: "bishop",
    1: "black-bishop",
    2: "black-king",
    3: "black-knight",
    4: "black-pawn",
    5: "black-queen",
    6: "black-rook",
    7: "white-bishop",
    8: "white-king",
    9: "white-knight",
    10: "white-pawn",
    11: "white-queen",
    12: "white-rook",
}

# Location of the images which contains sequential move images
image_folder = "test_selected"

# Location of the file of output result photos
output_folder = "marked"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Ordering the images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))))


for i in range(1, len(image_files), 2):
    image1_path = os.path.join(image_folder, image_files[i - 1])  # odd numered image (before)
    image2_path = os.path.join(image_folder, image_files[i])      # even numered image (after)

    # File check
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"Absence of file: {image1_path} or {image2_path}, this two will not be included.")
        continue


    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Applying YOLOv8 model
    results1 = model.predict(source=image1_path, save=True)
    results2 = model.predict(source=image2_path, save=True)

    # detected objects of two images
    detections1 = results1[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
    classes1 = results1[0].boxes.cls.cpu().numpy()  # Class IDs
    confidences1 = results1[0].boxes.conf.cpu().numpy()  # Accuracy

    detections2 = results2[0].boxes.xywh.cpu().numpy()
    classes2 = results2[0].boxes.cls.cpu().numpy()
    confidences2 = results2[0].boxes.conf.cpu().numpy()  # Accuracy

    # Accuracy filter
    confidence_threshold = 0.3
    filtered_detections1 = [(det, cls) for det, cls, conf in zip(detections1, classes1, confidences1) if conf > confidence_threshold]
    filtered_detections2 = [(det, cls) for det, cls, conf in zip(detections2, classes2, confidences2) if conf > confidence_threshold]

    # Controlling for all the classes from 1 to 12
    all_classes = set(range(1, 13))
    detected_classes1 = set(cls for _, cls in filtered_detections1)
    detected_classes2 = set(cls for _, cls in filtered_detections2)
    valid_classes = detected_classes1.union(detected_classes2)

    # matching objects between images
    moved_pieces = []
    coordinate_threshold = 50  # Pixel difference threshold

    for cls in valid_classes:
        class_detections1 = [det for det, det_cls in filtered_detections1 if det_cls == cls]
        class_detections2 = [det for det, det_cls in filtered_detections2 if det_cls == cls]

        if not class_detections1 or not class_detections2:
            continue  # If there is absence of a class in one of the image

        # difference matrix
        cost_matrix = np.zeros((len(class_detections1), len(class_detections2)))
        for i, det1 in enumerate(class_detections1):
            for j, det2 in enumerate(class_detections2):
                cost_matrix[i, j] = np.sqrt((det1[0] - det2[0]) ** 2 + (det1[1] - det2[1]) ** 2)  # Euclidean Distance

        # Matching with Hungarian Algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # comparing matched objects
        for row, col in zip(row_indices, col_indices):
            det1 = class_detections1[row]
            det2 = class_detections2[col]
            distance = cost_matrix[row, col]

            if distance > coordinate_threshold:  # distance threshold
                moved_pieces.append((cls, det1, det2))

    # marking on images
    y_position_image1 = 90
    y_position_image2 = 90
    line_height = 100

    for cls, det1, det2 in moved_pieces:
        x1, y1, w1, h1 = det1
        top_left1 = (int(x1 - w1 / 2), int(y1 - h1 / 2))
        bottom_right1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
        cv2.rectangle(image1, top_left1, bottom_right1, (0, 0, 255), 3)
        cv2.putText(
            image1,
            f"Moving: {class_names[cls]}",
            (10, y_position_image1),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (229, 237, 88),
            4,
        )

        x2, y2, w2, h2 = det2
        top_left2 = (int(x2 - w2 / 2), int(y2 - h2 / 2))
        bottom_right2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
        cv2.rectangle(image2, top_left2, bottom_right2, (0, 0, 255), 2)
        cv2.putText(
            image2,
            f"Moved: {class_names[cls]}",
            (10, y_position_image2),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (131, 40, 232),
            4,
        )

        # Arrows
        start_point1 = (int(x1), int(y1))
        end_point1 = (int(x2), int(y2))
        cv2.arrowedLine(image1, start_point1, end_point1, (255, 0, 0), 5, tipLength=0.08)

        start_point2 = (int(x2), int(y2))
        end_point2 = (int(x1), int(y1))
        cv2.arrowedLine(image2, end_point2, start_point2, (0, 0, 255), 5, tipLength=0.08)


        y_position_image1 += line_height
        y_position_image2 += line_height

    # Saving images to the path
    output1_path = os.path.join(output_folder, f"marked_{os.path.basename(image1_path)}")
    output2_path = os.path.join(output_folder, f"marked_{os.path.basename(image2_path)}")
    cv2.imwrite(output1_path, image1)
    cv2.imwrite(output2_path, image2)

    # Printing the results
    print(f"Moved Pieces ({image_files[i - 1]} -> {image_files[i]}):")
    for piece in moved_pieces:
        print(f"Class of the Piece: {piece[0]} ({class_names[piece[0]]})")
        print(f"Location Before: {piece[1]}")
        print(f"Location After: {piece[2]}")

print("All the images are processed and saved")
