import cv2
import numpy as np
import json
import os

class ClothingTracker:
    def __init__(self):
        self.captured_color = None
        self.lower_bound = None
        self.upper_bound = None
        self.target_lost_frames = 0
        self.last_position = None
        self.kalman = self.setup_kalman_filter()
        
        # Параметры для оценки расстояния
        self.focal_length = 600  # Примерное фокусное расстояние (нужна калибровка)
        self.real_person_width = 50  # Средняя ширина торса в см
        
        # История позиций для сглаживания траектории
        self.position_history = []
        self.max_history = 10
        
    def setup_kalman_filter(self):
        """Настройка фильтра Калмана для предсказания позиции"""
        # Модель состояния: [x, y, vx, vy]. Измерения: [x, y].
        kalman = cv2.KalmanFilter(4, 2)
        # Матрица измерений H: берем только позицию из состояния
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        # Переходная матрица F: добавляет скорость к позиции (dt=1 кадр)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        # Ковариация процессного шума Q: чем больше, тем гибче фильтр
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        return kalman
    
    def load_params(self, filename='clothing_params.json'):
        """Загрузка сохраненных параметров цвета"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                params = json.load(f)
                self.captured_color = np.array(params['captured_hsv'])
                self.lower_bound = np.array(params['lower_bound'])
                self.upper_bound = np.array(params['upper_bound'])
                return True
        return False
    
    def auto_capture_color(self, frame):
        """Автоматический захват цвета из центра кадра"""
        h, w = frame.shape[:2]
        # Берем центральную область
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        
        # Конвертируем в HSV
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Вычисляем средний цвет
        # Средний HSV по ROI — компромисс между простотой и устойчивостью
        mean_color = cv2.mean(roi_hsv)[:3]
        self.captured_color = np.array(mean_color, dtype=np.uint8)
        
        # Устанавливаем диапазоны (с правильными типами)
        self.lower_bound = np.array([
            max(0, int(self.captured_color[0]) - 15),
            max(0, int(self.captured_color[1]) - 50),
            max(0, int(self.captured_color[2]) - 50)
        ], dtype=np.uint8)
        self.upper_bound = np.array([
            min(179, int(self.captured_color[0]) + 15),
            min(255, int(self.captured_color[1]) + 50),
            min(255, int(self.captured_color[2]) + 50)
        ], dtype=np.uint8)
    
    def estimate_distance(self, bbox_width):
        """Оценка расстояния до человека на основе размера"""
        if bbox_width > 0:
            distance = (self.real_person_width * self.focal_length) / bbox_width
            return distance / 100  # Переводим в метры
        return 0
    
    def get_robot_coordinates(self, x_pixel, y_pixel, distance, frame_width, frame_height):
        """Преобразование пиксельных координат в систему координат робота"""
        # Центр изображения
        cx = frame_width / 2
        cy = frame_height / 2
        
        # Угол относительно центра камеры (в радианах)
        # Предполагаем угол обзора камеры примерно 60 градусов
        fov_horizontal = np.radians(60)
        fov_vertical = np.radians(45)
        
        # Нормируем смещение пикселя и масштабируем на FOV — получаем угловые отклонения
        angle_horizontal = ((x_pixel - cx) / frame_width) * fov_horizontal
        angle_vertical = ((cy - y_pixel) / frame_height) * fov_vertical
        
        # Координаты в системе робота
        x_robot = distance * np.sin(angle_horizontal)
        y_robot = distance * np.cos(angle_horizontal)
        theta = angle_horizontal
        
        return x_robot, y_robot, theta
    
    def process_frame(self, frame):
        """Основная обработка кадра"""
        # Конвертация в HSV и легкое размытие для снижения шума
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
        
        # Создаем маску по цвету
        mask = cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)
        
        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_found = False
        best_contour = None
        
        if contours:
            # Фильтруем контуры по размеру
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if valid_contours:
                if self.last_position is not None:
                    # Выбираем контур ближайший к последней позиции — стабилизирует трек
                    min_dist = float('inf')
                    for contour in valid_contours:
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            dist = np.sqrt((cx - self.last_position[0])**2 + 
                                         (cy - self.last_position[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_contour = contour
                else:
                    # Выбираем самый большой контур
                    best_contour = max(valid_contours, key=cv2.contourArea)
                
                if best_contour is not None:
                    target_found = True
        
        # Обработка результата
        result = {
            'frame': frame.copy(),
            'mask': mask,
            'target_found': target_found,
            'bbox': None,
            'center': None,
            'distance': None,
            'robot_coords': None
        }
        
        if target_found:
            # Сброс счетчика потери цели
            self.target_lost_frames = 0
            
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(best_contour)
            result['bbox'] = (x, y, w, h)
            
            # Центр объекта
            cx = x + w // 2
            cy = y + h // 2
            result['center'] = (cx, cy)
            
            # Обновляем фильтр Калмана
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            self.kalman.correct(measurement)
            prediction = self.kalman.predict()
            
            # Сохраняем позицию
            self.last_position = (int(prediction[0]), int(prediction[1]))
            
            # Добавляем в историю
            self.position_history.append(self.last_position)
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # Оценка расстояния
            distance = self.estimate_distance(w)
            result['distance'] = distance
            
            # Координаты в системе робота
            h_frame, w_frame = frame.shape[:2]
            robot_coords = self.get_robot_coordinates(cx, cy, distance, w_frame, h_frame)
            result['robot_coords'] = robot_coords
            
            # Рисуем на кадре
            cv2.drawContours(result['frame'], [best_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(result['frame'], (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(result['frame'], (cx, cy), 5, (0, 0, 255), -1)
            
            # Рисуем траекторию
            if len(self.position_history) > 1:
                for i in range(1, len(self.position_history)):
                    cv2.line(result['frame'], self.position_history[i-1], 
                            self.position_history[i], (255, 255, 0), 2)
            
            # Информация на экране
            info_text = [
                f"Distance: {distance:.2f} m",
                f"Robot coords: X={robot_coords[0]:.2f}, Y={robot_coords[1]:.2f}",
                f"Angle: {np.degrees(robot_coords[2]):.1f} deg"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(result['frame'], text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            # Цель потеряна
            self.target_lost_frames += 1
            
            # Используем предсказание Калмана если недавно потеряли
            if self.target_lost_frames < 30 and self.last_position is not None:
                prediction = self.kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                
                # Рисуем предполагаемую позицию
                cv2.circle(result['frame'], (pred_x, pred_y), 10, (0, 165, 255), 2)
                cv2.putText(result['frame'], "PREDICTED", (pred_x-30, pred_y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # Сброс если долго не видим цель
                self.last_position = None
                self.position_history = []
                cv2.putText(result['frame'], "TARGET LOST", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result


def main():
    tracker = ClothingTracker()
    
    # Пробуем загрузить сохраненные параметры
    if not tracker.load_params():
        print("Параметры не найдены. Используйте clothing_color_capture.py для настройки.")
        print("Или нажмите 'a' для автоматического захвата цвета из центра кадра.")
    
    cap = cv2.VideoCapture(0)
    
    print("\nУправление:")
    print("'a' - автоматический захват цвета из центра")
    print("'r' - сброс трекера")
    print("'q' - выход")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if tracker.captured_color is not None:
            # Обработка кадра
            result = tracker.process_frame(frame)
            
            # Отображение результатов
            cv2.imshow('Tracking Result', result['frame'])
            cv2.imshow('Color Mask', result['mask'])
            
            # Если есть координаты робота, выводим их
            if result['robot_coords'] is not None:
                x_robot, y_robot, theta = result['robot_coords']
                # Здесь можно отправить координаты в систему управления робота
                # send_to_robot(x_robot, y_robot, theta)
        else:
            cv2.putText(frame, "Press 'a' to auto-capture color", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Tracking Result', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            # Автозахват цвета
            tracker.auto_capture_color(frame)
            print(f"Цвет захвачен: HSV{tuple(tracker.captured_color)}")
            
        elif key == ord('r'):
            # Сброс трекера
            tracker.last_position = None
            tracker.position_history = []
            tracker.target_lost_frames = 0
            print("Трекер сброшен")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
