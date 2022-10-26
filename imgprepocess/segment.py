import cv2
import os
import numpy as np
import math
from collections import defaultdict
from PIL import Image
import face_recognition


# 人脸对齐
def align_face(image_array, landmarks):
    # 左右眼位置获取
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # 计算左右眼位置的平均点
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # 计算角度
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = math.atan2(dy, dx) * 180. / math.pi
    # 计算左右眼的中心
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # 按角度旋转图像
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


# 旋转图片中坐标
def rotate(origin, point, angle, row):
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


# 旋转图片中landmark
def rotate_landmarks(landmarks, eye_center, angle, row):
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


# 裁剪
def corp_face(image_array, landmarks):
    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 16 / 5
    bottom = lip_center[1] + mid_part * 7 / 5

    w = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    cropped_img = cv2.resize(cropped_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    return cropped_img, left, top


if __name__ == '__main__':
    for pic in os.listdir(r"./" + 'img'):
        image_array = cv2.imread(r"./" + 'img/' + pic)

        face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
        face_landmarks_dict = face_landmarks_list[0]

        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=face_landmarks_dict)
        Image.fromarray(np.hstack((image_array, aligned_face)))

        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                             eye_center=eye_center, angle=angle, row=image_array.shape[0])

        cropped_img, left, top = corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
        Image.fromarray(cropped_img)
        cv2.imwrite(os.path.join('out', pic), cropped_img)
        print(pic + '转化成功')
