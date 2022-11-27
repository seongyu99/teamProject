import numpy as np, cv2, time
import tensorflow as tf
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0)
model = tf.keras.models.load_model("./model")
capture = cv2.VideoCapture(0)

class_name = ["other","이선규", "임경수", "최정용"]
img_width, img_height = 180, 180

while True:
  success, image = capture.read()
  if not success: continue

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  detection_results = face_detection.process(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if detection_results.detections :
    for detection in detection_results.detections:
      location_data = detection.location_data
      if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
        bounding_box = location_data.relative_bounding_box
        x,y,w,h = int(bounding_box.xmin*image.shape[1]), int(bounding_box.ymin*image.shape[0]), \
                  int(bounding_box.width*image.shape[1]), int(bounding_box.height*image.shape[0])
        cv2.rectangle(image, (x, y, w, h), (0, 0, 255), 3)
        copy_image = image[y:y+h,x:x+w]
        copy_image = cv2.resize(image, (img_height,img_width))
        copy_image_array = tf.expand_dims(copy_image, 0)
        predict = model.predict(copy_image_array)
        score = tf.nn.softmax(predict[0])
        # if score<0:
        #   print("다른 사람")
        if class_name[np.argmax(score)] == class_name[1] or class_name[np.argmax(score)] == class_name[2] or \
                class_name[np.argmax(score)] == class_name[3]:
          cv2.putText(image,"Unlock",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
          cv2.imshow("test", image)
        else:
          cv2.putText(image, "Lock", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
          cv2.imshow("test", image)
  else:
    cv2.imshow("test", image)
  if cv2.waitKey(10)==ord(" "):
    break
capture.release()