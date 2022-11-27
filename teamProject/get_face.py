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
    model_selection=0, min_detection_confidence=0.5) as face_detection:  #얼굴 검출 클래스 선언
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("이미지 읽기 실패")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)  #얼굴 검출

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            detection = results.detections # 얼굴 검출 좌표를 포함한 클래스 정의(눈,코,입,귀의 중심점도 포함)

            location_data = detection.location_data
            if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
                bb = location_data.relative_bounding_box # 검출된 얼굴의 좌표값 참조하기
                x, y, w, h = int(bb.xmin*image.shape[1]), int(bb.ymin*image.shape[0]), \
                            int(bb.width*image.shape[1]), int(bb.height*image.shape[0])
                copy_image = image[y:y+h,x:x+w]  # 얼굴 인식된 이미지만 자르기
                copy_image = stretching(copy_image)
                count += 1
            cv2.imwrite("./faces/train/CJY/CJY"+str(count)+".jpg", copy_image)  # 검출된 얼굴 이미지를 잘라서 저장
            cv2.putText(copy_image, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(50) == ord(" ") or count == 100:
            break
cap.release()

