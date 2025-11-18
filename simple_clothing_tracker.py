import cv2
import numpy as np

def simple_clothing_tracker():
    """
    Простой трекер одежды с автоматическим захватом цвета
    Логика:
    - Пользователь размещает одежду в центре кадра.
    - По нажатию пробела измеряется средний HSV-цвет в ROI.
    - Создается маска по диапазону вокруг эталонного цвета, затем очищается морфологией.
    - Выбирается самый крупный контур, вычисляются центр, bbox, примерная дистанция и угол.
    - Ведется траектория и рисуется мини-радар «вид сверху» для интуитивной навигации.
    """
    cap = cv2.VideoCapture(0)
    
    # Переменные состояния
    tracking_enabled = False
    target_hsv = None
    hsv_range = (15, 50, 50)  # H, S, V диапазоны — ширина порога вокруг эталона
    
    # Для хранения истории движения
    trajectory = []
    max_trajectory_points = 50
    
    print("=== ПРОСТОЙ ТРЕКЕР ОДЕЖДЫ ===")
    print("ИНСТРУКЦИИ:")
    print("1. Поместите человека в центр кадра")
    print("2. Нажмите ПРОБЕЛ для захвата цвета одежды")
    print("3. Система начнет отслеживание")
    print("'c' - очистить траекторию")
    print("'q' - выход")
    print("="*30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_display = frame.copy()
        h, w = frame.shape[:2]
        
        # Рисуем центральную область для захвата
        # Квадратный ROI — компромисс между точностью и простотой взаимодействия.
        if not tracking_enabled:
            # Область захвата (центр кадра)
            roi_size = 100
            roi_x1 = w//2 - roi_size//2
            roi_y1 = h//2 - roi_size//2
            roi_x2 = roi_x1 + roi_size
            roi_y2 = roi_y1 + roi_size
            
            cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                         (0, 255, 255), 2)
            cv2.putText(frame_display, "Place clothing here", 
                       (roi_x1-20, roi_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_display, "Press SPACE to capture", 
                       (w//2-100, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Если трекинг включен
        if tracking_enabled and target_hsv is not None:
            # Конвертируем в HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Создаем маску по цвету (правильная работа с типами)
            # lower/upper ограничивают диапазон вокруг целевого HSV, все операции с int
            # и последующим приведение к uint8, чтобы избежать ошибок в OpenCV.
            lower = np.array([
                max(0, int(target_hsv[0]) - hsv_range[0]),
                max(0, int(target_hsv[1]) - hsv_range[1]),
                max(0, int(target_hsv[2]) - hsv_range[2])
            ], dtype=np.uint8)
            upper = np.array([
                min(179, int(target_hsv[0]) + hsv_range[0]),
                min(255, int(target_hsv[1]) + hsv_range[1]),
                min(255, int(target_hsv[2]) + hsv_range[2])
            ], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Очищаем маску
            # Простая морфология для удаления мелких шумов и расширения целевой области.
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Находим самый большой контур
                largest = max(contours, key=cv2.contourArea)
                
                # Проверяем минимальный размер
                # Слишком маленькие области игнорируются как шум/ложные срабатывания.
                if cv2.contourArea(largest) > 500:
                    # Получаем центр масс
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Добавляем в траекторию
                        trajectory.append((cx, cy))
                        if len(trajectory) > max_trajectory_points:
                            trajectory.pop(0)
                        
                        # Получаем bounding box
                        x, y, w_box, h_box = cv2.boundingRect(largest)
                        
                        # Оценка расстояния (простая формула)
                        # Предполагаем, что средняя ширина торса 50см
                        # Чем шире bbox, тем ближе объект.
                        focal_length = 600  # примерное значение
                        real_width = 50  # см
                        distance = (real_width * focal_length) / w_box if w_box > 0 else 0
                        distance_m = distance / 100
                        
                        # Угол до человека
                        # Вычислен по отклонению центра объекта от центра кадра.
                        angle = np.degrees(np.arctan((cx - w/2) / focal_length))
                        
                        # Координаты в системе робота
                        # Перевод из полярного представления (distance, angle) в XY.
                        x_robot = distance_m * np.sin(np.radians(angle))
                        y_robot = distance_m * np.cos(np.radians(angle))
                        
                        # Рисуем результаты
                        cv2.drawContours(frame_display, [largest], -1, (0, 255, 0), 2)
                        cv2.rectangle(frame_display, (x, y), (x+w_box, y+h_box), 
                                    (255, 0, 0), 2)
                        cv2.circle(frame_display, (cx, cy), 7, (0, 0, 255), -1)
                        
                        # Рисуем траекторию
                        # Градиент толщины для визуализации истории движения.
                        for i in range(1, len(trajectory)):
                            thickness = int(np.sqrt(i / float(len(trajectory))) * 5)
                            cv2.line(frame_display, trajectory[i-1], trajectory[i], 
                                   (255, 255, 0), thickness)
                        
                        # Выводим информацию
                        info = [
                            f"Distance: {distance_m:.2f} m",
                            f"Angle: {angle:.1f} deg",
                            f"Robot X: {x_robot:.2f} m",
                            f"Robot Y: {y_robot:.2f} m",
                            f"Target HSV: {target_hsv}"
                        ]
                        
                        for i, text in enumerate(info):
                            cv2.putText(frame_display, text, (10, 25*(i+1)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Рисуем "радар" - вид сверху
                        # Простая мини-карта, где центр — робот, точка — целевой объект.
                        radar_center = (w - 100, h - 100)
                        radar_radius = 80
                        cv2.circle(frame_display, radar_center, radar_radius, (100, 100, 100), 1)
                        cv2.circle(frame_display, radar_center, radar_radius//2, (100, 100, 100), 1)
                        
                        # Позиция человека на радаре
                        if distance_m < 5:  # Максимум 5 метров на радаре
                            radar_x = int(radar_center[0] + (x_robot/5) * radar_radius)
                            radar_y = int(radar_center[1] - (y_robot/5) * radar_radius)
                            cv2.circle(frame_display, (radar_x, radar_y), 5, (0, 255, 0), -1)
                        
                        # Робот в центре радара
                        cv2.circle(frame_display, radar_center, 3, (255, 0, 0), -1)
                        cv2.putText(frame_display, "TOP VIEW", 
                                  (radar_center[0]-30, radar_center[1]+radar_radius+15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    else:
                        cv2.putText(frame_display, "Invalid contour", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_display, "Target too small", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(frame_display, "No target found", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Показываем маску
            # Уменьшенная копия бинарной маски для удобства диагностики.
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_display = cv2.resize(mask_colored, (w//3, h//3))
            frame_display[10:10+h//3, w-w//3-10:w-10] = mask_display
            cv2.putText(frame_display, "MASK", (w-w//3, h//3+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Clothing Tracker', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Захват цвета из центральной области
            if not tracking_enabled:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Вычисляем средний цвет в ROI
                # Среднее по ROI устойчиво, но при сложном фоне лучше адаптивные методики.
                target_hsv = cv2.mean(hsv_roi)[:3]
                target_hsv = tuple(map(int, target_hsv))
                
                tracking_enabled = True
                print(f"Tracking started! Target HSV: {target_hsv}")
                
        elif key == ord('c'):
            # Очистить траекторию
            trajectory = []
            print("Trajectory cleared")
            
        elif key == ord('r'):
            # Сброс трекинга
            tracking_enabled = False
            target_hsv = None
            trajectory = []
            print("Tracking reset")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simple_clothing_tracker()
