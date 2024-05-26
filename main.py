import cv2
import torch
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta
import time
import ftplib
import os
import threading
import wave
import pyaudio

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 라벨을 한글로 변환하는 사전
label_dict = {
    'person': '사람',
    'bicycle': '자전거',
    'car': '자동차',
    'motorcycle': '오토바이',
    'airplane': '비행기',
    'bus': '버스',
    'train': '기차',
    'truck': '트럭',
    'boat': '보트',
    'traffic light': '신호등',
    'fire hydrant': '소화전',
    'stop sign': '정지 신호',
    'parking meter': '주차 요금기',
    'bench': '벤치',
    'bird': '새',
    'cat': '고양이',
    'dog': '개',
    'horse': '말',
    'sheep': '양',
    'cow': '소',
    'elephant': '코끼리',
    'bear': '곰',
    'zebra': '얼룩말',
    'giraffe': '기린',
    'backpack': '배낭',
    'umbrella': '우산',
    'handbag': '핸드백',
    'tie': '넥타이',
    'suitcase': '여행가방',
    'frisbee': '프리스비',
    'skis': '스키',
    'snowboard': '스노보드',
    'sports ball': '스포츠 공',
    'kite': '연',
    'baseball bat': '야구 배트',
    'baseball glove': '야구 글러브',
    'skateboard': '스케이트보드',
    'surfboard': '서핑보드',
    'tennis racket': '테니스 라켓',
    'bottle': '병',
    'wine glass': '와인 잔',
    'cup': '컵',
    'fork': '포크',
    'knife': '나이프',
    'spoon': '숟가락',
    'bowl': '그릇',
    'banana': '바나나',
    'apple': '사과',
    'sandwich': '샌드위치',
    'orange': '오렌지',
    'broccoli': '브로콜리',
    'carrot': '당근',
    'hot dog': '핫도그',
    'pizza': '피자',
    'donut': '도넛',
    'cake': '케이크',
    'chair': '의자',
    'couch': '소파',
    'potted plant': '화분',
    'bed': '침대',
    'dining table': '식탁',
    'toilet': '화장실',
    'tv': 'TV',
    'laptop': '노트북',
    'mouse': '마우스',
    'remote': '리모컨',
    'keyboard': '키보드',
    'cell phone': '휴대전화',
    'microwave': '전자레인지',
    'oven': '오븐',
    'toaster': '토스터',
    'sink': '싱크대',
    'refrigerator': '냉장고',
    'book': '책',
    'clock': '시계',
    'vase': '꽃병',
    'scissors': '가위',
    'teddy bear': '테디 베어',
    'hair drier': '헤어 드라이어',
    'toothbrush': '칫솔'
}

# 한글 폰트 로드 (Mac에 기본 설치된 "AppleGothic" 폰트 사용)
font_path = "/System/Library/Fonts/AppleGothic.ttf"
font_size = 32
font = ImageFont.truetype(font_path, font_size)

# FTP 서버 정보
FTP_SERVER = "서버 주소 입력"
FTP_USERNAME = "아이디 입력"
FTP_PASSWORD = "패스워드 입력"
FTP_UPLOAD_PATH = "서버 경로 입력"

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 비디오 코덱 설정 및 파일 저장 초기화 (MP4 형식, H.264 코덱)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()

def record_audio(file_path, stop_event):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio recording saved to {file_path}")

def upload_to_ftp(file_path, ftp_server, ftp_username, ftp_password, ftp_upload_path):
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(ftp_username, ftp_password)
            with open(file_path, 'rb') as file:
                ftp.storbinary(f'STOR {ftp_upload_path}/{os.path.basename(file_path)}', file)
        print(f"Uploaded {file_path} to FTP server.")
        os.remove(file_path)  # 업로드 후 파일 삭제
        print(f"Deleted local file: {file_path}")
    except Exception as e:
        print(f"Error uploading {file_path} to FTP server: {e}")

# 1분 간격으로 새로운 비디오 파일 생성 및 저장
start_time = datetime.now()
video_writer = None
video_file_path = None
audio_file_path = None
stop_event = threading.Event()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 모델에 프레임 전달하여 객체 인식 수행
    results = model(frame)

    # 결과 얻기
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # 프레임 크기 얻기
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    # 현재 시간
    now = datetime.now()
    elapsed_time = now - start_time

    # 1분 경과 시 새로운 비디오 파일 생성
    if elapsed_time >= timedelta(minutes=1):
        if video_writer is not None:
            video_writer.release()
            stop_event.set()  # 오디오 녹음 중지
            audio_thread.join()
            upload_to_ftp(video_file_path, FTP_SERVER, FTP_USERNAME, FTP_PASSWORD, FTP_UPLOAD_PATH)
            upload_to_ftp(audio_file_path, FTP_SERVER, FTP_USERNAME, FTP_PASSWORD, FTP_UPLOAD_PATH)

        start_time = now
        detected_objects = [label_dict[model.names[int(label)]] for label in labels]
        detected_objects_str = "_".join(detected_objects) if detected_objects else "no_detection"
        video_file_path = f"{now.strftime('%Y%m%d_%H%M%S')}_{detected_objects_str}.mp4"
        audio_file_path = f"{now.strftime('%Y%m%d_%H%M%S')}_{detected_objects_str}.wav"
        video_writer = cv2.VideoWriter(video_file_path, fourcc, 7.0, (x_shape, y_shape))
        stop_event.clear()
        audio_thread = threading.Thread(target=record_audio, args=(audio_file_path, stop_event))
        audio_thread.start()
        print(f"Started new video file: {video_file_path}")

    # PIL 이미지를 사용하여 한글 라벨 그리기
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 인식된 객체의 경계 상자와 라벨 그리기
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:  # 신뢰도 임계값 설정
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            label = model.names[int(labels[i])]
            label_ko = label_dict.get(label, label)  # 한글 라벨 얻기
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            for offset in range(-1, 2):
                draw.text((x1 + offset, y1 - font_size - 10), label_ko, font=font, fill="green")
                draw.text((x1, y1 - font_size - 10 + offset), label_ko, font=font, fill="green")

    # OpenCV 이미지를 다시 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 비디오 프레임 작성
    if video_writer is not None:
        video_writer.write(frame)

    # 프레임 보여주기
    cv2.imshow('YOLOv5 Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
if video_writer is not None:
    video_writer.release()
    stop_event.set()
    audio_thread.join()
    upload_to_ftp(video_file_path, FTP_SERVER, FTP_USERNAME, FTP_PASSWORD, FTP_UPLOAD_PATH)
    upload_to_ftp(audio_file_path, FTP_SERVER, FTP_USERNAME, FTP_PASSWORD, FTP_UPLOAD_PATH)
audio.terminate()
cv2.destroyAllWindows()
