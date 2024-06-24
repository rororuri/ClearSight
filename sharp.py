from PIL import Image, ImageFilter
import os

# 입력 폴더와 출력 폴더 경로 설정
input_folder = 'A'  # 입력 이미지 폴더 경로를 지정하세요
output_folder = 'B'  # 출력 이미지 폴더 경로를 지정하세요

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 입력 폴더 내의 모든 이미지 파일에 대해 샤프닝 필터 적용
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        try:
            # 이미지 열기
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # 샤프닝 필터 적용
            sharpened_image = image.filter(ImageFilter.SHARPEN)

            # 결과 이미지 저장
            output_path = os.path.join(output_folder, filename)
            sharpened_image.save(output_path)

            print(f'Sharpened and saved: {output_path}')
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')
