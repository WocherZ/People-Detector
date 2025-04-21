import cv2
import streamlit as st
from ultralytics import YOLO
import time
from PIL import Image
import numpy as np

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")  # Можно использовать 'yolov8s.pt', 'yolov8m.pt' и т.д. для большей точности

# Настройка Streamlit
st.title("Подсчёт людей с камеры с помощью YOLOv8")
st.write("Приложение определяет количество людей на видео с камеры и отмечает их bounding boxes.")

# Выбор камеры
camera_option = st.selectbox("Выберите камеру:", options=[0, 1, 2, 3], index=0)

# Инициализация видеозахвата
cap = cv2.VideoCapture(camera_option)
if not cap.isOpened():
    st.error("Не удалось открыть камеру. Попробуйте другую.")
    st.stop()

# Создание элементов интерфейса
image_placeholder = st.empty()
count_placeholder = st.empty()

# Основной цикл обработки
last_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Не удалось получить кадр с камеры.")
        break
    
    current_time = time.time()
    if current_time - last_time >= 1.0:  # Обрабатываем кадр раз в секунду
        last_time = current_time
        
        # Детекция с помощью YOLO
        results = model.predict(frame, classes=[0], verbose=False)  # class 0 - это 'person' в COCO
        
        # Визуализация результатов
        annotated_frame = results[0].plot()
        
        # Конвертация цветов (OpenCV -> PIL)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_frame_rgb)
        
        # Подсчёт людей
        num_people = len(results[0].boxes)
        
        # Обновление интерфейса
        image_placeholder.image(pil_image, caption="Текущий кадр с камеры", use_column_width=True)
        count_placeholder.success(f"Количество людей в кадре: {num_people}")
    
    # Небольшая задержка для снижения нагрузки на CPU
    time.sleep(0.1)

# Освобождение ресурсов (этот код выполнится только если цикл прервётся)
cap.release()