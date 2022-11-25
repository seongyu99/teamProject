import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
count = 0
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        # image = cv2.imread("faces/test/mom.jpg", cv2.IMREAD_COLOR)
        # success = 1
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                # 얼굴 검출 좌표 참조
                location_data = detection.location_data
                if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
                    bb = location_data.relative_bounding_box
                    bb_box = [
                    bb.xmin, bb.ymin,
                    bb.width, bb.height,
                    ]
                    x, y, w, h = int(bb.xmin*image.shape[1]), int(bb.ymin*image.shape[0]), \
                                int(bb.width*image.shape[1]), int(bb.height*image.shape[0])
                    copy_image = image[y:y+h,x:x+w]  # 얼굴 인식된 이미지만 자르기
                    # copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
                    count += 1
                # cv2.imwrite("./faces/train/Dad/Dad"+str(count)+".jpg", copy_image)
                cv2.putText(copy_image, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Flip the image horizontally for a selfie-view display.
                # cv2.imshow('MediaPipe Face Detection', cv2.flip(copy_image, 1))
                cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
                if cv2.waitKey(50) == 27 or count == 300:
                    break
cap.release()