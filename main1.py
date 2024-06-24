import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# YOLO 모델 로드 및 설정
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 실제 이미지 객체 탐지 함수
def detect_objects(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 파일을 읽을 수 없습니다: {image_path}")
        return None, None

    img = cv2.resize(img, None, fx=1.1, fy=1.1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    confidence_threshold = 0.3

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]  # 해당 객체의 탐지 확률
            detected_objects.append((label, x, y, x + w, y + h, confidence))  # 정확도를 함께 저장

    return img, detected_objects

# 객체 인식된 이미지를 넣는 리스트
fake_image_list = [] # 이미지 값이 들어가는 리스트 FAKE
fake_object_list = [] # 이미지의 오브젝트 값이 들어가는 리스트 FAKE
real_image_list = [] # 동일 하지만 GAN 돌리고 난 뒤 사진
real_object_list = [] # 동일 하지만 GAN 돌리고 난 뒤의 오브젝트 값

# 이미지 리스트를 가져옵니다
fake_image_name_list = os.listdir("C:/Users/SERVER/Desktop/hello/photo_Before_GAN") # GAN들리지 않은 사진들의 파일 이름들이 저장되는 리스트
real_image_name_list = os.listdir("C:/Users/SERVER/Desktop/hello/photo_After_GAN") # GAN돌린 뒤의 resize를 마친 사진들의 파일 이름들이 저장되는 리스트 

# 실제 이미지와 가짜 이미지 경로
real_image_root ="photo_After_GAN/" # 사진들이 있는 폴더까지의 경로
fake_image_root ="photo_Before_GAN/" # 사진들이 있는 폴더까지의 경로

# 실제 이미지와 가짜 이미지를 객체 탐지 함수에 입력하여 객체 정보를 얻습니다.
for fn in real_image_name_list: # fn은 파일 이름, GAN돌리고 난 뒤의 사진 사용
    real_image, real_objects = detect_objects(real_image_root + fn) # 해당 경로에 있는 사진으로 사진안에 있는 오브젝트 감지
    real_image_list.append(real_image) # image값을 이미지 리스트에 순서대로 넣는다
    real_object_list.append(real_objects) # object값을 object 리스트에 순서대로 넣는다

for fn in fake_image_name_list: # 위와 동일 하지만 fake사진을 활용
    fake_image, fake_objects = detect_objects(fake_image_root + fn)
    fake_image_list.append(fake_image)
    fake_object_list.append(fake_objects)

# calculate_mAP 함수 수정
def calculate_mAP(real_objects, fake_objects):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 실제 이미지 객체 탐지 평가
    for realObj in real_object_list:
        for real_object in realObj:
            label, x1, y1, x2, y2, confidence = real_object
            is_detected = False
            for fakeObj in fake_object_list:
                for fake_object in fakeObj:
                    if label == fake_object[0]:
                        x1_f, y1_f, x2_f, y2_f, confidence_f = fake_object[1:]
                        # 겹치는 영역 계산
                        intersection_x1 = max(x1, x1_f)
                        intersection_y1 = max(y1, y1_f)
                        intersection_x2 = min(x2, x2_f)
                        intersection_y2 = min(y2, y2_f)
                        # 영역이 양수이면 객체가 겹침
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            is_detected = True
                            break
                if is_detected:
                    true_positives += 1
                else:
                    false_negatives += 1

    false_objects = []

    # 가짜 이미지 객체 탐지 평가
    for fakeObj in fake_object_list:
        for fake_object in fakeObj:
            label, x1_f, y1_f, x2_f, y2_f, confidence_f = fake_object
            is_detected = False
            for realObj in real_object_list:
                for real_object in realObj:
                    if label == real_object[0]:
                        x1, y1, x2, y2, confidence = real_object[1:]
                        # 겹치는 영역 계산
                        intersection_x1 = max(x1, x1_f)
                        intersection_y1 = max(y1, y1_f)
                        intersection_x2 = min(x2, x2_f)
                        intersection_y2 = min(y2, y2_f)
                        # 영역이 양수이면 객체가 겹침
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            is_detected = True
                            break
                if not is_detected:
                    false_objects.append(fake_object)
                else:
                    true_positives += 1

    
# 결과 이미지에 객체 표시 (기존 코드와 동일)
for i in range(len(real_object_list)):
    for label, x1, y1, x2, y2, confidence in real_object_list[i]:
        cv2.rectangle(real_image_list[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"  # 클래스 이름과 정확도 표시
        cv2.putText(real_image_list[i], text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

for i in range(len(fake_object_list)):
    for label, x1, y1, x2, y2, confidence in fake_object_list[i]:
        cv2.rectangle(fake_image_list[i], (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{label}: {confidence:.2f}"  # 클래스 이름과 정확도 표시
        cv2.putText(fake_image_list[i], text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 결과 이미지 표시 (기존 코드와 동일)
# cv2.imshow("Real Image", real_image)
# cv2.imshow("Fake Image", fake_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

i = 0
for rImg in real_image_list: # 실질적으로 이미지 파일을 저장하는 함수 GAN돌린 사진 저장
    print(real_image_name_list[i])
    cv2.imwrite("C:/Users/SERVER/Desktop/hello/A" + "/" + real_image_name_list[i],rImg)
    i += 1

i = 0
for fImg in fake_image_list: # 위와 동일 하지만 FAKE사진 사용해서 저장
    cv2.imwrite("C:/Users/SERVER/Desktop/hello/B" + "/" + fake_image_name_list[i], fImg)
    i += 1