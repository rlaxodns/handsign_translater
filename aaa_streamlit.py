# https://biz.typecast.ai/org/overview  <= 타입캐스트 API 

import cv2
import mediapipe as mp
import numpy as np
import torch
import requests
import json
import time
import os
import io

from functions.model import TimeSeriesTransformer
from PIL import ImageFont, ImageDraw, Image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import sounddevice as sd
from scipy.io.wavfile import read
# from playsound import playsound  # TTS 음성 파일 재생을 위해 추가

import streamlit as st

from langchain.chat_models import ChatOpenAI

import warnings
warnings.filterwarnings('ignore')


def convert_text_to_speech(text, api_key, character_id):
    """
    Convert text to speech using Typecast API 텍스트를 Typecast API를 사용하여 음성으로 변환하는 함수.
    
    Args:
    text (str): 음성으로 변환할 텍스트
    api_key (str): Typecast API의 인증 키
    character_id (str): Typecast에서 사용할 캐릭터의 ID
    
    기능:
    이 함수는 Typecast API를 호출하여 텍스트를 음성으로 변환하고, 변환된 음성을 재생합니다.
    """
    
    # API 요청 헤더 설정 (Authorization: Bearer 토큰 방식으로 인증)
    HEADERS = {'Authorization': f'Bearer {api_key}'}

    try:
        # 1. 텍스트 음성 변환 요청 (POST 요청)yt
        # 'text'와 'character_id'를 포함하여 API에 음성 변환 요청을 보냅니다.
        r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
            'text': text,   # 음성으로 변환할 텍스트
            'lang': 'auto', # 자동 언어 감지
            'actor_id': character_id, # 사용할 캐릭터의 ID
            'xapi_hd': True, # 일부 파라미터 설정 (화질 등을 포함한 고급 옵션)
            'model_version': 'latest' # 최신 모델 버전 사용
        })

        # 요청 성공 시 응답에서 음성 파일 URL을 가져옵니다.
        speak_url = r.json()['result']['speak_v2_url']

         # 2. 음성 변환 작업이 완료될 때까지 폴링(polling)
        for _ in range(60): # 최대 60번 반복 (1분 동안 기다림)
            # 변환된 음성의 상태를 확인하기 위해 GET 요청
            r = requests.get(speak_url, headers=HEADERS)

            # API 응답에서 결과를 추출
            ret = r.json()['result']

            # 음성 변환이 완료된 경우
            if ret['status'] == 'done':
                # 3. 음성 다운로드 링크를 통해 음성 데이터를 가져옵니다.
                audio_data = requests.get(ret['audio_download_url']).content  # 바이너리 데이터로 다운로드
                
                # 다운로드한 음성 데이터를 메모리 내에서 스트림으로 변환
                audio_stream = io.BytesIO(audio_data)  

                # 4. 음성 파일을 읽어 들여 샘플레이트와 오디오 데이터를 얻습니다.
                sample_rate, audio = read(audio_stream)  

                # 5. 음성 재생 (sounddevice를 사용하여 오디오 재생)
                sd.play(audio, samplerate=sample_rate)  # 오디오 재생
                sd.wait()  # 오디오가 끝날 때까지 대기

                break # 음성 변환이 완료되면 루프 종료

    except Exception as e:
        # 예외 발생 시 오류 메시지 출력
        print(f"TTS conversion error: {e}")


# Typecast API Configuration
TYPECAST_API_KEY = ""
TYPECAST_CHARACTER_ID = "622964d6255364be41659078"
# "66d000ee0742c43c93a0ada1" 남자 목소리 : 도현
# "65bb3a1976b69213594357fc" 여자 목소리 : 진서

def speak(text):
    # Authorization 헤더를 사용하여 Typecast API 인증을 위한 API 키 설정
    headers = {'Authorization': f'Bearer {TYPECAST_API_KEY}'}
    
    # POST 요청을 보내 텍스트를 음성으로 변환하는 요청을 Typecast API로 전송
    r = requests.post('https://typecast.ai/api/speak', headers=headers, json={
        'text': text,   # 변환할 텍스트를 전달
        'lang': 'auto', # 자동 언어 감지
        'actor_id': TYPECAST_CHARACTER_ID,   # 사용할 음성 배우(캐릭터) ID 설정 
        'xapi_hd': True,    # 고해상도 오디오 요청
        'model_version': 'latest'  # 최신 모델 사용
    })
    
    # API 응답에서 음성 합성 URL 추출
    speak_url = r.json()['result']['speak_v2_url']

     # 음성 합성 결과가 완료될 때까지 60번까지 확인하는 반복문
    for _ in range(60):
        # 음성 합성 상태를 확인하기 위해 GET 요청
        r = requests.get(speak_url, headers=headers)
        ret = r.json()['result'] # 응답에서 음성 합성 상태 정보 추출

        # 음성 합성 상태가 완료되었으면
        if ret['status'] == 'done':
            # 음성 파일을 다운로드하기 위한 URL에서 오디오 데이터를 받아옴
            audio_data = requests.get(ret['audio_download_url']).content
            # 오디오 데이터를 메모리 상에서 스트림으로 변환
            audio_stream = io.BytesIO(audio_data)  

            # 오디오 스트림에서 샘플링 주파수와 오디오 데이터를 읽어옴
            sample_rate, audio = read(audio_stream)  

            # 오디오를 재생
            sd.play(audio, samplerate=sample_rate)  
            
            # 오디오가 끝날 때까지 기다림
            sd.wait()  

            # 음성 재생 후 루프 종료
            break

# Model and API Initialization
HOME_PATH = 'C:/ai5/Bon_project/main_project/custom-final-custom'
last_added_time = 0
openai_api_key = ''

# LLM 설정
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
        "4. 단어의 문맥적 의미를 고려하여 자연스럽게 사용하세요.\n"
        "5. 리스트 내의 단어로 자연스러운 문장을 출력할 수 없다면, '-'로 출력하세요\n"
        "단어 리스트: {filtered_words}\n"
        "출력 예시:\n"
        "- 입력: ['안녕', '만나다', '반갑다']\n"
        "- 출력: '안녕 만나서, 반가워'\n"
        "- 입력: ['반갑다', '나', '소개']\n"
        "- 출력: '내 소개를 할께!'\n"
        "- 입력: ['나', '이름', '지혜']\n"
        "- 출력: '내 이름은 지혜야.'\n"
        "- 입력: ['나', '태운', '이름', '태운']\n"
        "- 출력: '내 이름은 태운이야.'\n"
        "- 입력: ['나', '이름', '기현']\n"
        "- 출력: '내 이름은 기현이야.  '\n"
        "- 입력: ['나', '초밥', '좋아해']\n"
        "- 출력: '나는 초밥을 좋아해.'\n"
        "- 입력: ['어제', '파스타', '먹다']\n"
        "- 출력: '어제 파스타를 먹었어.'\n"
        "- 입력: ['나', '책', '읽다', '좋아하다']\n"
        "- 출력: '나는 책을 읽는 걸 좋아해.'\n"
        "- 입력: ['나', '책', '읽다', '좋아하다']\n"
        "- 출력: '나는 책을 읽는 걸 좋아해.'\n"
        "- 입력: ['나', '요리', '좋아해']\n"
        "- 출력: '나는 요리도 좋아해.'\n"
        "- 입력: ['고마워']\n"
        "- 출력: '고마워!'\n"
        "- 입력: ['다음', '다시', '만나다']\n"
        "- 출력: '다음에 다시 만나!'\n"
        "리스트에 포함되지 않은 단어를 추가하거나 문장을 초월해서 말하지 마세요."
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

# Model 설정 및 불러오기
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 37
model = TimeSeriesTransformer(224, 64, 8, 1, 37, 30, 1)
# model_weights_path = f'{HOME_PATH}/models/model_final_custom2.pt'
model_weights_path = 'C:/Users/kim/Desktop/대나무팀/custom-final/models/model_final_custom4.pt'
model = torch.load(model_weights_path, map_location=device)

# 0공백, 1안녕, 2만나다, 3반갑다, 4나, 5입니다, 6이름, 7지혜, 8기현, 9태운, 
# 10좋아해, 11초밥, 12파스타, 13책, 14요리, 15감사합니다, 16다음, 17다시, 18소개, 19요즘
# 20수화, 21배우다, 22컴퓨터, 23열심히, 24노력, 25중이다, 26무얼, 27생각, 28어제, 29먹다, 
# 30맛있다, 31이유, 32읽다, 33듣다, 34너, 35치킨, 36대학원

gesture = {0: "", 1: "안녕", 2: "만나다", 3: "반갑다", 4: "나", 5: "입니다", 6: "이름", 7: "지혜", 8: "기현", 9: "태운", 
           10: "좋아해", 11 : "초밥", 12 : "파스타", 13 : "책", 14: "요리", 15 : "고마워", 16 : "다음", 17 : "다시", 18: "소개", 19: "요즘",
           20: "수화", 21: "배우다", 22: "컴퓨터", 23: "열심히", 24: "노력", 25:"중이다", 26: "무얼", 27: "생각", 28: "어제", 29: "먹다",
           30: "맛있다", 31: "이유", 32: "읽다", 33: "듣다", 34: "너", 35 : "치킨", 36: "대학원"}

actions = ["", "안녕", "만나다", "반갑다", "나", "입니다", "이름", "지혜", "기현", "태운", 
           "좋아해", "초밥", "파스타", "책", "요리", "고마워", "다음", "다시", "소개", "요즘",
           "수화", "배우다", "컴퓨터", "열심히", "노력", "중이다", "무얼", "생각", "어제", "먹다",
           "맛있다", "이유", "읽다", "듣다", "너", "치킨", "대학원"]

seq_length = 30


st.set_page_config(page_title="수화 번역", page_icon='🙌')
st.markdown("<h1 style='text-align: center;'> 수화 번역 </h1>", unsafe_allow_html=True)


    

# MediaPipe 설정 및 초기화 추가
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)    # "http://192.168.0.76:8080/video"

cap.set(cv2.CAP_PROP_FPS, 20) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

seq = []
action_seq = []
if 'lang' not in st.session_state:
    st.session_state.lang = []

font = ImageFont.truetype("malgunbd.ttf", 40)

frame_placeholder = st.empty()
text_placeholder = st.empty()
text_placeholder2 = st.empty()
state = False

if st.button('확인'):
    state = True

def add_word(word):
    st.session_state.lang.append(word)

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        print("카메라에 접근할 수 없습니다.")
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # cv2.imshow('img', img)

    # key = cv2.waitKey(1)

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

    # 시퀀스가 충분히 쌓였는지 확인 (길이가 seq_length와 동일하거나 그 이상인지 확인)
    if len(seq) < seq_length:
        continue
    
    # seq의 마지막 seq_length만큼 슬라이싱하여 모델 입력으로 사용
    input_data = np.array(seq[-seq_length:], dtype=np.float32)  # (seq_length, feature_dim)
    
    # 여기서 3차원 텐서로 변경하기 위해 batch 차원을 추가함
    input_data = np.expand_dims(input_data, axis=0)  # (1, seq_length, feature_dim)
    
    # 텐서를 Torch 텐서로 변환하고, 장치에 맞게 올림
    input_data = torch.FloatTensor(input_data).to(device)  # 최종 형태: (batch_size, seq_length, feature_dim)

    # 모델 예측 
    y_pred = model(input_data) # (batch_size, num_classes)
    
    # 최대 확률값과 그 인덱스를 가져옴
    conf, idx = torch.max(y_pred.data, dim=1)
    
    if conf < 0.25:
        continue
    
    # 예측된 동작을 가져옴
    action = actions[idx.item()]
    action_seq.append(action)

    if len(action_seq) < 10:
        continue
    
    this_action = ''

    # 이 조건문은 action_seq 리스트의 마지막 5개 요소가 모두 같은지 확인
    if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5]:
         this_action = action
    
    # 마지막 5개의 액션이 동일하면 이를 this_action으로 사용
    if action_seq[-5:] == [action] * 5:
        if time.time() - last_added_time > 2:
             if action and (not st.session_state.lang or st.session_state.lang[-1] != action):  # 빈 문자열이 아닌 경우에만 추가
                # st.session_state.lang.append(action)
                add_word(action)
                last_added_time = time.time()  # 마지막 추가 시간을 갱신하여 중복 추가 방지
                print(f"Added to st.session_state.lang: {action}")
                text_placeholder2.info(f'추가된 전체 단어 : {st.session_state.lang}')
                print(st.session_state.lang)
    
    # 이미지에 현재 액션 표시
    if action != '':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 30), f'[{action}]', font=font, fill=(255, 0, 0))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    frame_placeholder.image(img, channels='BGR')

    # key = cv2.waitKey(1)
    # 시퀀스가 너무 길면 초기화
    if len(seq) >= 300:
        seq = []

    if len(action_seq) >= 300:
        action_seq = []

    # cv2.imshow('img', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

    # # 백스페이스 키로 마지막 액션 제거
    # if key == 8:
    #     if st.session_state.lang:
    #         st.session_state.lang.pop()
    #         print("삭제", st.session_state.lang)
        
    # print('lang 확인용 :', st.session_state.lang)

     # 스페이스 키로 현재 단어 리스트로부터 문장 생성 및 음성 출력
    if state:
        # state = False
        # cap.release()

        print('st.session_state.lang : ', st.session_state.lang)
        # filtered_words = [word for word in st.session_state.lang if word.strip()]
        filtered_words = st.session_state.lang
        print('filtered_words :', filtered_words)

        if filtered_words:
            # result_text = chain.run(filtered_words=" ".join(filtered_words))
            result_text = chain.run(filtered_words)
            print("출력:", result_text)
            
            # '출력:' 이후의 문장만 TTS로 변환하여 읽어줌
            output_text = result_text.split(': ')[-1].strip().strip("'")
            # output_text = result_text.strip().strip("'")
            text_placeholder.info(f'문장 : {output_text}')
            convert_text_to_speech(output_text, TYPECAST_API_KEY, TYPECAST_CHARACTER_ID)
            # speak(output_text)
            
            st.session_state.lang = []
            state = False
            st.rerun()
        else:
            print("생성할 문장이 없습니다.")
            state = False
            st.session_state.lang = []
            st.rerun()

       
            
    #     if key == ord('q'):
    #         break
    
    # if key == ord(' '):  # 스페이스바 눌렀을 때 문장 생성 및 TTS 실행
    #     st.session_state.lang = []

    # if key == 8:  # 백스페이스 키를 눌렀을 때 마지막 단어 삭제
    #     if st.session_state.lang:
    #         st.session_state.lang.pop()
    #         print("삭제", st.session_state.lang)

    if result.right_hand_landmarks is None and result.left_hand_landmarks is None and result.pose_landmarks is None:
        continue
