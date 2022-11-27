import cv2, time
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def search_value_idx(hist, bias = 0):
    for i in range(hist.shape[0]):
        idx = np.abs(bias - i)                     # 검색 위치 (처음 또는 마지막)
        if hist[idx] > 0:  return idx                             # 위치 반환
    return -1
def stretching(image):
    bsize, ranges = [64], [0, 256]  # 계급 개수 및 화소 범위
    hist = cv2.calcHist([image], [0], None, bsize, ranges)

    bin_width = ranges[1] / bsize[0]  # 계급 너비
    high = search_value_idx(hist, bsize[0] - 1) * bin_width
    low = search_value_idx(hist, 0) * bin_width

    idx = np.arange(0, 256)
    idx = (idx - low) * 255 / (high - low)  # 수식 적용하여 인덱스 생성
    idx[0:int(low)] = 0
    idx[int(high + 1):] = 255

    dst = cv2.LUT(image, idx.astype('uint8'))
    return dst

cap = cv2.VideoCapture(0)
count = 0
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        # image.flags.writeable = True
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
                    dst = stretching(copy_image)
                    # copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
                    count += 1
                cv2.imwrite("./faces/other/CJY/CJY"+str(count)+".jpg", dst)
                cv2.putText(dst, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Flip the image horizontally for a selfie-view display.
                # cv2.imshow('MediaPipe Face Detection', cv2.flip(copy_image, 1))
                cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(50) == ord(" ") or count == 100:
            break
# cap.release()

