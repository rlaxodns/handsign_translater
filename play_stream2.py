import cv2
import mediapipe as mp
import numpy as np
import torch
import requests
import time
import io
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import read
from PIL import ImageFont, ImageDraw, Image
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from functions.model import TimeSeriesTransformer

# ===== TTS 함수 =====
def convert_text_to_speech(text, api_key, character_id):
    HEADERS = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
            'text': text,
            'lang': 'auto',
            'actor_id': character_id,
            'xapi_hd': True,
            'model_version': 'latest'
        })
        speak_url = r.json()['result']['speak_v2_url']

        for _ in range(60):
            r = requests.get(speak_url, headers=HEADERS)
            ret = r.json()['result']
            if ret['status'] == 'done':
                audio_data = requests.get(ret['audio_download_url']).content
                audio_stream = io.BytesIO(audio_data)
                sample_rate, audio = read(audio_stream)
                sd.play(audio, samplerate=sample_rate)
                sd.wait()
                break
    except Exception as e:
        print(f"TTS conversion error: {e}")

def speak(text, api_key, character_id):
    HEADERS = {'Authorization': f'Bearer {api_key}'}
    r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
        'text': text,
        'lang': 'auto',
        'actor_id': character_id,
        'xapi_hd': True,
        'model_version': 'latest'
    })
    speak_url = r.json()['result']['speak_v2_url']

    for _ in range(60):
        r = requests.get(speak_url, headers=HEADERS)
        ret = r.json()['result']
        if ret['status'] == 'done':
            audio_data = requests.get(ret['audio_download_url']).content
            audio_stream = io.BytesIO(audio_data)
            sample_rate, audio = read(audio_stream)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
            break

# ===== 설정 =====
TYPECAST_API_KEY = ""
TYPECAST_CHARACTER_ID = "622964d6255364be41659078"
openai_api_key = 'sk-...'  # 실제키로 변경
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

# 모델 로드
model = TimeSeriesTransformer(224, 64, 8, 1, 37, 30, 1)
model_weights_path = r'C:\Users\kim\Desktop\대나무팀\custom-final\models\model_final_custom4.pt'
model = torch.load(model_weights_path, map_location=device)
model.eval()

gesture = {0: "", 1: "안녕", 2: "만나다", 3: "반갑다", 4: "나", 5: "입니다", 6: "이름", 7: "지혜", 8: "기현", 9: "태운",
           10: "좋아해", 11: "초밥", 12: "파스타", 13: "책", 14: "요리", 15: "고마워", 16: "다음", 17: "다시", 18: "소개", 19: "요즘",
           20: "수화", 21: "배우다", 22: "컴퓨터", 23: "열심히", 24: "노력", 25:"중이다", 26: "무얼", 27: "생각", 28: "어제", 29: "먹다",
           30: "맛있다", 31: "이유", 32: "읽다", 33: "듣다", 34: "너", 35: "치킨", 36: "대학원"}

actions = ["", "안녕", "만나다", "반갑다", "나", "입니다", "이름", "지혜", "기현", "태운",
           "좋아해", "초밥", "파스타", "책", "요리", "고마워", "다음", "다시", "소개", "요즘",
           "수화", "배우다", "컴퓨터", "열심히", "노력", "중이다", "무얼", "생각", "어제", "먹다",
           "맛있다", "이유", "읽다", "듣다", "너", "치킨", "대학원"]

seq_length = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
font = ImageFont.truetype("malgunbd.ttf", 40)

# 세션 상태 관리
if "seq" not in st.session_state:
    st.session_state.seq = []
if "action_seq" not in st.session_state:
    st.session_state.action_seq = []
if "lang" not in st.session_state:
    st.session_state.lang = []
if "last_added_time" not in st.session_state:
    st.session_state.last_added_time = 0


st.set_page_config(page_title="수어 인식 Demo", layout="wide")
st.title("📹 실시간 수어 인식 Demo")

# WebRTC 설정
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 비디오 처리용 클래스
class VideoTransformer:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calc_hand_angle(self, hand_landmarks):
        if hand_landmarks is None:
            return np.zeros((99,))
        joint = np.zeros((21,4))
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
        v = v2 - v1
        # 0으로 나누는 경우 회피
        norm = np.linalg.norm(v, axis=1)
        norm[norm==0] = 1e-6
        v = v / norm[:, np.newaxis]

        angle = np.arccos(
            np.einsum(
                "nt,nt->n",
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]
            )
        )
        angle = np.degrees(angle)
        angle = np.concatenate([joint.flatten(), angle])
        return angle

    def calc_pose_angle(self, pose_landmarks):
        if pose_landmarks is None:
            return np.zeros((27,))
        joint = np.zeros((6,4))
        idxs = [11,12,13,14,23,24]
        k = 0
        for j, lm in enumerate(pose_landmarks.landmark):
            if j in idxs:
                joint[k] = [lm.x, lm.y, lm.z, lm.visibility]
                k+=1

        v1 = joint[[0,1,1,0,0,4],:]
        v2 = joint[[1,3,5,2,4,5],:]
        norm = np.linalg.norm(v2 - v1, axis=1)
        norm[norm==0] = 1e-6
        v = (v2 - v1) / norm[:, np.newaxis]
        angle = np.arccos(np.einsum("nt,nt->n", v[[0,1],:], v[[2,3],:]))
        angle = np.degrees(angle)
        angle = np.concatenate([joint.flatten(), angle])
        return angle

    def transform(self, frame):
        # frame: av.VideoFrame
        img = frame.to_ndarray(format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(img_rgb)

        # drawing landmarks
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # 특징 추출
        right = self.calc_hand_angle(result.right_hand_landmarks)
        left = self.calc_hand_angle(result.left_hand_landmarks)

        pose = self.calc_pose_angle(result.pose_landmarks)
        data = np.concatenate([right, left, pose])
        st.session_state.seq.append(data)

        if len(st.session_state.seq) >= seq_length:
            input_data = np.array(st.session_state.seq[-seq_length:], dtype=np.float32)
            input_data = torch.tensor(input_data, dtype=torch.float32, device=device).unsqueeze(0)
            y_pred = model(input_data)
            conf, idx = torch.max(y_pred.data, dim=1)
            if conf > 0.25:
                action = actions[idx.item()]
                st.session_state.action_seq.append(action)
                if len(st.session_state.action_seq) >= 10:
                    if st.session_state.action_seq[-5:] == [action]*5:
                        if time.time() - st.session_state.last_added_time > 2:
                            if action and (not st.session_state.lang or st.session_state.lang[-1] != action):
                                st.session_state.lang.append(action)
                                st.session_state.last_added_time = time.time()

        # 단어 리스트 화면에 표시
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        current_text = " ".join(st.session_state.lang) if st.session_state.lang else "인식된 단어 없음"
        draw.text((10, 30), f'{current_text}', font=font, fill=(255, 0, 0))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img

# WebRTC 스트리머
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=VideoTransformer,
    async_transform=True
)

col1, col2 = st.columns(2)

with col2:
    if st.button("문장 생성 및 읽기"):
        filtered_words = [w for w in st.session_state.lang if w.strip()]
        if filtered_words:
            result_text = chain.run(filtered_words=" ".join(filtered_words))
            st.write("생성 문장:", result_text)
            output_text = result_text.split(': ')[-1].strip().strip("'")
            convert_text_to_speech(output_text, TYPECAST_API_KEY, TYPECAST_CHARACTER_ID)
            speak(output_text, TYPECAST_API_KEY, TYPECAST_CHARACTER_ID)
        else:
            st.write("생성할 문장이 없습니다.")
        st.session_state.lang = []

    if st.button("마지막 단어 삭제"):
        if st.session_state.lang:
            st.session_state.lang.pop()
            st.write("삭제 후 단어 리스트:", st.session_state.lang)
