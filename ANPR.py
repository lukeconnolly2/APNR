from ultralytics import YOLO
import cv2
import easyocr
from difflib import SequenceMatcher

DETECT_EVERY_N_FRAMES = 1

license_plate_detector = YOLO("best.pt")
reader = easyocr.Reader(['en'])
accepted_license_plates = ['08-WX-TEST', '08-WX-TEST2']


def crop_license_plate(image, box):
    return image[box[1]:box[3], box[0]:box[2]]


def process_license_plate(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


video_stream = cv2.VideoCapture(0)
frame_count = 0
while True:
    cropped_license_plate = None
    license_plate_text = ''
    ret, frame = video_stream.read()
    frame_count += 1
    if ret and frame_count % DETECT_EVERY_N_FRAMES == 0:
        license_plate_results = license_plate_detector(frame, verbose=False)[0]
        for box in license_plate_results.boxes.xyxy.tolist():
            box = [int(x) for x in box]
            cropped_license_plate = crop_license_plate(frame, box)
            processed_license_plate = cv2.cvtColor(cropped_license_plate, cv2.COLOR_BGR2GRAY)
            if processed_license_plate is not None:
                license_plate_text = reader.readtext(processed_license_plate, detail=0)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if len(license_plate_text) > 0:
                cv2.putText(frame, license_plate_text[-1], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (255, 0, 0), 5, cv2.LINE_AA)
                if any([similar(license_plate_text[-1], accepted_license_plate) > 0.8 for accepted_license_plate in
                        accepted_license_plates]):
                    print(f'License plate detected! - {license_plate_text[-1]}')
                    cv2.imwrite('license_plate.png', cropped_license_plate)
                    cv2.putText(frame, 'Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 5, cv2.LINE_AA)
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
