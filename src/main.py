import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.models import Detection, Segmentation, Gun, PersonType, Person, PixelLocation, GunType
from src.config import get_settings

SETTINGS = get_settings()


app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)

@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array

def load_image(file: UploadFile) -> np.ndarray:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid image format")
    
    return np.array(img_obj)

@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> Segmentation:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold)
    return segmentation

@app.post("/annotate_people")
def annotate_people(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> Response:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)
    
    img_pil = Image.fromarray(annotated_img)
    img_io = io.BytesIO()
    img_pil.save(img_io, format="JPEG")
    img_io.seek(0)
    
    return Response(content=img_io.read(), media_type="image/jpeg")

@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> dict:
    img_array = load_image(file)
    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    
    return {
        "detection": detection,
        "segmentation": segmentation
    }

@app.post("/annotate")
def annotate(
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> Response:
    image_array = load_image(file)
    image_detection = detector.detect_guns(image_array, threshold)
    image_segmentation = detector.segment_people(image_array, threshold)

    annotated_image = annotate_detection(image_array, image_detection)
    annotated_image = annotate_segmentation(annotated_image, image_segmentation, draw_boxes)

    image_pil = Image.fromarray(annotated_image)
    image_io = io.BytesIO()
    image_pil.save(image_io, format="JPEG")
    image_io.seek(0)

    return Response(content=image_io.read(), media_type="image/jpeg")

def create_gun(label, box):
    x_center = (box[0] + box[2]) // 2
    y_center = (box[1] + box[3]) // 2
    return Gun(
        gun_type=label,
        location=PixelLocation(x=x_center, y=y_center)
    )

@app.post("/guns")
def guns(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> list[Gun]:
    img_array = load_image(file)
    detection = detector.detect_guns(img_array, threshold)
    
    guns = []
    
    detection.labels = list(map(lambda label: label.lower(), detection.labels))
    
    is_valid_gun_type = lambda label: label in [GunType.pistol, GunType.rifle]
    
    guns = list(
        map(
            lambda label_box: create_gun(*label_box),
            filter(
                lambda label_box: is_valid_gun_type(label_box[0]),
                zip(detection.labels, detection.boxes)
            )
        )
    )

    return guns

def create_person_data_from_image(args):
    label, polygon = args
    polygon_points = np.array(polygon)
    x_center = int(np.mean(polygon_points[:, 0]))
    y_center = int(np.mean(polygon_points[:, 1]))
    area = int(cv2.contourArea(polygon_points))
    
    person_type = PersonType(label)
    pixel_location = PixelLocation(x=x_center, y=y_center)
    
    return Person(
        person_type=person_type,
        location=pixel_location,
        area=area
    )

@app.post("/people")
def people(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    detector: GunDetector = Depends(get_gun_detector)
) -> list[Person]:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold)
    
    segmentation.labels = list(map(lambda label: label.lower(), segmentation.labels))
    
    people = list(map(create_person_data_from_image, zip(segmentation.labels, segmentation.polygons)))
    
    return people

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", port=8000, host="127.0.0.1")
