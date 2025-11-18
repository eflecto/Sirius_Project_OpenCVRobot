#!/usr/bin/env python3
"""
Минимальный тестовый скрипт для быстрой проверки детектирования по цвету
"""

import cv2
import numpy as np

print("МИНИМАЛЬНЫЙ ТЕСТ ДЕТЕКТОРА")
print("1. Наведите камеру на цветную одежду")
print("2. Нажмите любую клавишу мыши для захвата цвета")
print("3. ESC для выхода")
print("-"*40)

cap = cv2.VideoCapture(0)
target_color = None
mouse_x, mouse_y = 320, 240

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, target_color
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Захват цвета в точке клика
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_color = hsv[y, x]
        print(f"Цвет захвачен в точке ({x},{y}): HSV = {target_color}")

cv2.namedWindow('Test Detector')
cv2.setMouseCallback('Test Detector', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    
    # Показываем курсор
    cv2.circle(display, (mouse_x, mouse_y), 5, (255, 255, 0), 2)
    
    if target_color is not None:
        # Создаем маску
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Широкие диапазоны для теста
        lower = np.array([
            max(0, int(target_color[0]) - 20),
            max(0, int(target_color[1]) - 60),
            max(0, int(target_color[2]) - 60)
        ], dtype=np.uint8)
        
        upper = np.array([
            min(179, int(target_color[0]) + 20),
            min(255, int(target_color[1]) + 60),
            min(255, int(target_color[2]) + 60)
        ], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Очистка маски
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Рисуем все контуры
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
            
            # Находим самый большой
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Показываем площадь
            area = cv2.contourArea(largest)
            cv2.putText(display, f"Area: {area:.0f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Показываем маску в углу
        mask_small = cv2.resize(mask, (160, 120))
        display[10:130, 10:170] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        
        # Информация о цвете
        cv2.putText(display, f"Target HSV: {target_color}", (10, display.shape[0]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"Range: [{lower}, {upper}]", (10, display.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(display, "Click on clothing to capture color", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Test Detector', display)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset
        target_color = None
        print("Reset")

cap.release()
cv2.destroyAllWindows()
