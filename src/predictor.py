from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()

def calculate_distance(bbox: list[int], segment_polygon: Polygon) -> float:
    min_x, min_y, max_x, max_y = bbox
    bbox_polygon = box(min_x, min_y, max_x, max_y)
    return segment_polygon.distance(bbox_polygon)

def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    segment_polygon = Polygon([(point[0], point[1]) for point in segment])
    
    distances_gotten = [
        calculate_distance(bbox, segment_polygon) for bbox in bboxes
    ]
    
    small_distances = [
        (distance, bbox) for distance, bbox in zip(distances_gotten, bboxes) if distance <= max_distance
    ] or []
    
    if len(small_distances) == 0:
        return None
    
    small_distances.sort(key=lambda x: x[0])
    
    return small_distances[0][1]

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img

def fill_polygon_and_draw_box(img_to_annotate_segmentation: np.ndarray, label: str, polygon: list[list[int]], box: list[int], draw_boxes: bool = True) -> np.ndarray:
    red_color = (255, 0, 0)
    green_color = (0, 255, 0)
    color = red_color if label == "danger" else green_color

    pts = np.array(list(map(lambda coord: [int(coord[0]), int(coord[1])], polygon)), np.int32).reshape((-1, 1, 2))

    alpha = 0.5
    alpha = 1 - alpha

    overlay = img_to_annotate_segmentation.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img_to_annotate_segmentation, alpha, 0, img_to_annotate_segmentation)

    if draw_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_to_annotate_segmentation, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img=img_to_annotate_segmentation,
            text=label,
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=color,
            thickness=2,
        )

    return img_to_annotate_segmentation

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    img_to_annotate_segmentation = image_array.copy()

    draw_boxes_list = [draw_boxes] * len(segmentation.labels)

    list(map(lambda args: fill_polygon_and_draw_box(*args),
                zip(
                    [img_to_annotate_segmentation] * len(segmentation.labels),
                    segmentation.labels, 
                    segmentation.polygons,
                    segmentation.boxes,
                    draw_boxes_list,
                )
            ))

    return img_to_annotate_segmentation


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
        
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 20):
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()

        person_indexes = list(filter(lambda i: labels[i] == 0, range(len(labels))))

        person_boxes = list(
            map(lambda i: [int(v) for v in results.boxes.xyxy[i].tolist()], person_indexes)
        )

        person_polygons = list(
            map(lambda i: [[int(coord[0]), int(coord[1])] for coord in results.masks.xy[i]], person_indexes)
        )

        guns_detections = self.detect_guns(image_array, threshold)

        person_labels = list(
            map(
                lambda person_polygon: "danger" if match_gun_bbox(person_polygon, guns_detections.boxes, max_distance) else "safe",
                person_polygons
            )
        )

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(person_boxes),
            polygons=person_polygons,
            boxes=person_boxes,
            labels=person_labels
        )
