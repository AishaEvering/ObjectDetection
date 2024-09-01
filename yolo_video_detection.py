from ultralytics import YOLO
import cv2
import math

MODEL = "predictions/content/runs/detect/train3/weights/best.pt"
CLASSNAMES = ["afro", "bantu knots", "bob", "braids",
              "cornrows", "fade", "locs", "long", "sisterlocs", "twa"]


def video_detection(path: str, classNames: str = CLASSNAMES, model_path: str = MODEL):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model = YOLO(model_path)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        # out.write(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get rectangle
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # get confidence
                conf = math.ceil((box.conf[0] * 100))/100

                # get class id and class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # set the label
                label = f'{class_name} {conf}'

                # get the size of the label
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                # create label rectangle
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [
                              255, 0, 255], -1, cv2.LINE_AA)  # filled

                cv2.putText(img, label, (x1, y1-2), 0, 1,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        yield img

        # out.write(img)
        # cv2.imshow("Image", img)
        # if cv2.waitKey(1) & 0xFF == ord('1'):
        #     break

    # out.release()
    cv2.destroyAllWindows()
