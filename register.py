import dlib
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# 얼굴 인식 모델 로드
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# 얼굴 랜드마크 모델 로드
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 사전에 등록된 얼굴 임베딩과 라벨
known_embeddings = []
known_labels = []
blink_counts = []


# 사전에 등록된 얼굴 임베딩과 라벨 추가 함수
def add_known_face_embedding(embedding, label):
    known_embeddings.append(embedding)
    known_labels.append(label)


# 실시간 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 얼굴 인식을 위한 Dlib face detector 생성
face_detector = dlib.get_frontal_face_detector()

# Person 1의 얼굴 임베딩 등록
images = ['person1.jpg', 'person2.jpg']
for image in images:
    person2_image = cv2.imread(image)
    person2_dlib_frame = cv2.cvtColor(person2_image, cv2.COLOR_BGR2RGB)
    person2_faces = face_detector(person2_dlib_frame)
    person2_landmarks = landmark_detector(person2_dlib_frame, person2_faces[0])
    person2_embedding = face_recognition_model.compute_face_descriptor(person2_dlib_frame, person2_landmarks)
    label = image.split('0')[0]
    add_known_face_embedding(person2_embedding, label)
    blink_counts.append(0)

# 감정 분석을 위한 모델 로드
emotion_model = load_model('emotion_model.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def calculate_eye_aspect_ratio(eye_landmarks):
    eye_landmarks = np.array([[p.x, p.y] for p in eye_landmarks])

    horizontal_dist1 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    horizontal_dist2 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[2])
    horizontal_length = (horizontal_dist1 + horizontal_dist2) / 2

    vertical_dist1 = np.linalg.norm(eye_landmarks[5] - eye_landmarks[1])
    vertical_dist2 = np.linalg.norm(eye_landmarks[4] - eye_landmarks[2])
    vertical_length = (vertical_dist1 + vertical_dist2) / 2

    aspect_ratio = vertical_length / horizontal_length

    return aspect_ratio


while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    # 이미지를 Dlib 형식으로 변환
    dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 인식 수행
    faces = face_detector(dlib_frame)

    # 각 얼굴에 대해 얼굴 식별 및 감정 분석 수행
    for face in faces:
        # 얼굴 랜드마크 추출
        landmarks = landmark_detector(dlib_frame, face)
        # 얼굴 임베딩 추출
        embedding = face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)

        # 등록된 얼굴 임베딩과 비교하여 누구인지 판별
        distances = []
        for known_embedding in known_embeddings:
            distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
            distances.append(distance)

        # 판별 결과 출력
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        if min_distance <= 0.4:
            label = known_labels[min_distance_idx]
        else:
            label = 'Unknown'

        # 얼굴 영역에 라벨 표시
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {min_distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 얼굴 영역 추출
        face_img = frame[top:bottom, left:right]

        # 감정 분석을 위해 이미지를 흑백으로 변환
        gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # 이미지를 모델에 전달하여 감정 예측
        resized_img = cv2.resize(gray_face_img, (64, 64), interpolation=cv2.INTER_AREA)
        img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255
        predictions = emotion_model.predict(img_array)
        max_index = np.argmax(predictions[0])
        emotion = emotion_labels[max_index]
        emotions = {k: v for k, v in zip(emotion_labels, predictions[0])}
        # 감정 결과 출력
        cv2.putText(frame, emotion, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        i = 25
        for e, v in emotions.items():
            v = format(v, '.4f')
            cv2.putText(frame, f"{e}: {v}", (left, bottom + i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            i += 25

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_landmarks = landmark_detector(gray, face)

        # 눈의 랜드마크 인덱스
        left_eye_landmarks = [gray_landmarks.part(i) for i in range(36, 42)]
        right_eye_landmarks = [gray_landmarks.part(i) for i in range(42, 48)]

        # 눈의 종횡비 계산
        left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye_landmarks)
        right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye_landmarks)

        # 눈 깜빡임 확인
        if left_eye_aspect_ratio < 0.3 and right_eye_aspect_ratio < 0.3:
            blink_counts[min_distance_idx] += 1

        # 눈 영역 표시
        for point in left_eye_landmarks:
            cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)
        for point in right_eye_landmarks:
            cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)
        cv2.putText(frame, f"Blinks: {blink_counts[min_distance_idx]}", (left, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
    # 화면에 출력
    cv2.imshow('Face Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 및 창 종료
cap.release()
cv2.destroyAllWindows()