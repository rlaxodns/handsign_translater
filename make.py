import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os

HOME_PATH = r'C:\ai5\Bon_project\main_project\custom-final'

actions = ["만나다", '반갑다', '안녕', "어디", "가", "학교", "잘", "다음", "다시", "나", "",
            "너", "열심히", "닭", "대화", "멀다", "아니", "가깝다", "끝나다", "좋아하다"] 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5               
)

i = 19                             
# 0만나다, 1반갑다, 2안녕, 3 어디, 4가, 5학교, 6잘, 7다음, 8다시, 9나, 10공란, 11너, 12열심히, 13닭, 14대화, 15멀다
# 16아니, 17가깝다, 18끝나다, 19좋아하다

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

created_time = int(time.time())

default_array = np.array(range(225), dtype='float64')

while cap.isOpened():
    seq = []

    status, frame = cap.read()

    if frame is None:
        break

    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = holistic.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow("Dataset", frame)

    if cv2.waitKey(1) == ord("q"):
        break

    data = []

    if result.right_hand_landmarks is None and result.left_hand_landmarks is None and result.pose_landmarks is None:
        continue

    right_joint_angle = np.array([])

    if result.right_hand_landmarks is not None:
        joint = np.zeros((21, 4))

        for j, lm in enumerate(result.right_hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

        v = v2 - v1

        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(
            np.einsum(
                "nt,nt->n",
                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
            )
        )

        angle = np.degrees(angle)
        angle = np.array([angle], dtype=np.float32)

        right_joint_angle = np.concatenate([joint.flatten(), angle.reshape(angle.shape[1],)])
    else:
        right_joint_angle = np.zeros((99, ))

    left_joint_angle = np.array([])

    if result.left_hand_landmarks is not None:
        joint = np.zeros((21, 4))

        for j, lm in enumerate(result.left_hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

        v = v2 - v1

        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(
            np.einsum(
                "nt,nt->n",
                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
            )
        )

        angle = np.degrees(angle)
        angle = np.array([angle], dtype=np.float32)

        left_joint_angle = np.concatenate([joint.flatten(), angle.reshape(angle.shape[1],)])
    else:
        left_joint_angle = np.zeros((99, ))

    pose_joint_angle = np.array([])

    if result.pose_landmarks is not None:
        joint = np.zeros((6, 4))

        k = 0

        for j, lm in enumerate(result.pose_landmarks.landmark):
            if j in [11, 12, 13, 14, 23, 24]:
                joint[k] = [lm.x, lm.y, lm.z, lm.visibility]

                k += 1

        v1 = joint[[0, 1, 1, 0, 0, 4], :]
        v2 = joint[[1, 3, 5, 2, 4, 5], :]

        v = v2 - v1

        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(
            np.einsum(
                "nt,nt->n",
                v[[0, 1], :],
                v[[2, 3], :],
            )
        )

        angle = np.degrees(angle)
        angle = np.array([angle], dtype=np.float32)

        pose_joint_angle = np.concatenate([joint.flatten(), angle.reshape(angle.shape[1],), np.array([i])])
    else:
        pose_joint_angle = np.zeros((27, ))

    data = np.concatenate([data, right_joint_angle, left_joint_angle, pose_joint_angle])

    default_array = np.vstack((default_array, data))

if actions[i] != '':
    os.makedirs(f'{HOME_PATH}/dataset/{actions[i]}', exist_ok=True)

    pd.DataFrame(default_array).iloc[1:, :].to_csv(f'{HOME_PATH}/dataset/{actions[i]}/raw_{actions[i]}_{created_time}.csv', header=False, index=False)
else:
    pd.DataFrame(default_array).iloc[1:, :].to_csv(f'{HOME_PATH}/dataset/raw_{actions[i]}_{created_time}.csv', header=False, index=False)