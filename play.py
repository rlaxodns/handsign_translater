import cv2
import mediapipe as mp
import numpy as np


import torch

from functions.model import TimeSeriesTransformer
from PIL import ImageFont, ImageDraw, Image
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
HOME_PATH = r'C:\ai5\Bon_project\main_project\custom-final'
last_added_time = 0
openai_api_key = ''

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    api_key=openai_api_key,
)


prompt = PromptTemplate(
    input_variables=["filtered_words"],
    template=(
        "리스트에 포함되지 않은 단어를 새롭게 생성하거나 문장을 새롭게 생성하지 마세요."
        "아래의 단어들만 사용해서 문어체 문장을 작성하세요. "
        "반드시 리스트에 있는 단어만 사용하고, 필요 없는 단어는 삭제하세요. "
        "출력은 다음 조건을 따라야 합니다:\n"
        "1. 문장은 한 문장만 작성하세요.\n"
        "2. 출력 문장은 마침표(.)로 끝나야 합니다.\n"
        "3. 구어체나 부적절한 표현은 사용하지 마세요.\n"
        "4. 단어의 문맥적 의미를 고려하여 자연스럽게 사용하세요.\n\n"
        "5. 리스트 내의 단어로 자연스러운 문장을 출력할 수 없다면, '-'로 출력하세요\n\n"
        "단어 리스트: {filtered_words}\n\n"
        "출력 예시:\n"
        "- 입력: ['안녕', '만나다', '어디', '반갑다']\n"
        "- 출력: '안녕 만나서, 반가워'\n"
        "- 입력: ['안녕', '학교', '가깝다']\n"
        "- 출력: '안녕, 학교는 가깝다.'\n\n"
        "- 원래 리스트: ['', '안녕', '열심히', '안녕', '끝나다', '', '닭']\n\n"
        "- 입력: ['안녕', '열심히', '끝나다']\n\n"
        "- 출력: '안녕, 열심히 끝내야지.\n\n'"
        "- 원래 리스트: ['', '열심히', '끝나다', '']\n\n"
        "- 출력: '열심히 끝내야지.'\n\n"
        "리스트에 포함되지 않은 단어를 추가하거나 문장을 초월 번역하지 마세요."
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 18 

model = TimeSeriesTransformer(224, 64, 8, 1, 10, 30, 1)

model_weights_path = 'models\model_final.pt'

model = torch.load(model_weights_path, map_location=device)

gesture = {0: "만나다", 1: "반갑다", 2: "안녕", 3: "어디", 4: "가", 5: "학교", 6: "잘", 7: "다음", 8: "다시", 9: "나", 10:"", 11:"너", 12:"열심히", 13: "닭",14: "대화", 15: "멀다", 16:"아니", 17: "가깝다", 18:"끝나다", 19:"공부"}
actions = ["만나다", "반갑다", "안녕", "어디", "가", "학교", "잘", "다음", "다시", "나", "", "너", "열심히", "닭", "대화", "멀다", "아니", "가깝다", "끝나다", "공부"]

seq_length = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
seq = [] 
action_seq = []
lang = []
font = ImageFont.truetype("malgunbd.ttf", 40)

while cap.isOpened():
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('img', img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

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

        pose_joint_angle = np.concatenate([joint.flatten(), angle.reshape(angle.shape[1],)])
    else:
        pose_joint_angle = np.zeros((27, ))

    data = np.concatenate([right_joint_angle, left_joint_angle, pose_joint_angle])

    seq.append(data)

    if len(seq) < seq_length:
        continue

    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

    input_data = torch.FloatTensor(input_data).to(device)

    y_pred = model(input_data)

    conf, idx = torch.max(y_pred.data, dim=1, keepdim=True)

    # if idx != 10:
        # print(conf)

    if conf < 0.25:
        continue

    action = actions[idx]

    action_seq.append(action)

    if len(action_seq) < 10:
        continue

    this_action = ''

    # if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5]:

    # if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == \
    #    action_seq[-6] == action_seq[-7] == action_seq[-8] == action_seq[-9] == action_seq[-10]:

    if len(set(action_seq[-10:])) == 1:
        this_action = action

        if time.time() - last_added_time > 2:
            if len(lang) == 0 or lang[-1] != this_action:
                lang.append(this_action)  # 중복된 제스처를 추가하지 않음
                print(f"Added to lang: {this_action}")

    if action != '':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_img)

        draw.text((10, 30), f'[{action}]', font=font, fill=(255, 0, 0))

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)                    

    if len(seq) >= 300:
        seq = []

    if len(action_seq) >= 300:
        action_seq = []

    cv2.imshow('img', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    if key == 8:
        lang.pop()
        print("삭제", lang)
        
    if key == 32:
        print("원래 리스트:", lang)

        filtered_words = [word for word in lang if word.strip()]

        if filtered_words:
            result = chain.run(filtered_words=" ".join(filtered_words))

            print("생성된 문어체 문장:")

            print(result)
        else:
            print("생성할 문장이 없습니다.")

        lang = []

        if key == ord('q'):
           break

cap.release()

cv2.destroyAllWindows()