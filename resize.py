import cv2
import os

path = "C:/Users/SERVER/Desktop/hello/results" # 한글경로 X
file_name = os.listdir(path) # path는 폴더까지의 경로이며 해당 줄의 코드는 폴더 내부에 있는 모든 파일들을 읽는 것이다
# 사진 이외의 것이 들어가면 안됨
for fn in file_name:
    print(path + '/' + fn)
    IMG = cv2.imread(path + '/' + fn, cv2.IMREAD_COLOR) # 이미지 읽기
    resized_IMG = cv2.resize(IMG, (777, 391)) # 이미지 사이즈 바꾸기 777 X 391으로 지정함 나중에 필요에 따라 바꾸면 됨
    cv2.imwrite("photo_After_GAN/" + fn, resized_IMG) # 이미지 저장 이미지 이름은 읽어올떄와 동일하게 함