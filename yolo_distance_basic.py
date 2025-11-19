"""
Базовый скрипт определения расстояния до людей с YOLO8
Без коррекции дисторсии (добавим позже)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time


class PersonDistanceDetector:
    """Базовый детектор людей с определением расстояния и угла"""
    
    def __init__(self, calibration_file='camera_calib_result.npz', model_path='yolov8n.pt'):
        """
        Инициализация детектора
        
        Args:
            calibration_file: путь к файлу калибровки камеры (.npz)
            model_path: путь к модели YOLO (yolov8n.pt, yolov8s.pt, и т.д.)
        """
        print("=" * 70)
        print("ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРА")
        print("=" * 70)
        
        # Загружаем калибровку камеры
        print(f"\n1. Загрузка калибровки из {calibration_file}...")
        calib_data = np.load(calibration_file)
        
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']
        
        # Извлекаем параметры камеры
        self.fx = self.camera_matrix[0, 0]  # Фокусное расстояние по X
        self.fy = self.camera_matrix[1, 1]  # Фокусное расстояние по Y
        self.cx = self.camera_matrix[0, 2]  # Оптический центр X
        self.cy = self.camera_matrix[1, 2]  # Оптический центр Y
        
        print(f"   Фокусное расстояние: fx={self.fx:.2f}, fy={self.fy:.2f}")
        print(f"   Оптический центр: cx={self.cx:.2f}, cy={self.cy:.2f}")
        
        # Средние параметры человека (в сантиметрах)
        self.PERSON_HEIGHT = 170  # см - средний рост человека
        self.PERSON_WIDTH = 45    # см - средняя ширина плеч
        
        print(f"\n2. Параметры человека:")
        print(f"   Рост: {self.PERSON_HEIGHT} см")
        print(f"   Ширина плеч: {self.PERSON_WIDTH} см")
        
        # Загружаем модель YOLO
        print(f"\n3. Загрузка модели YOLO8 ({model_path})...")
        self.model = YOLO(model_path)
        self.person_class_id = 0  # ID класса "person" в COCO dataset
        
        print("\n✓ Инициализация завершена!")
        print("=" * 70)
    
    def calculate_distance_and_angle(self, bbox):
        """
        Вычисляет расстояние до человека и угол относительно центра камеры
        
        Args:
            bbox: [x1, y1, x2, y2] - координаты bounding box
            
        Returns:
            dict с результатами:
                - distance_z: расстояние вдоль оси камеры (м)
                - distance_real: реальное расстояние с учётом угла (м)
                - angle_deg: угол относительно центра (градусы)
                - offset_x: горизонтальное смещение (м)
                - dist_by_width: расстояние по ширине (м)
                - dist_by_height: расстояние по высоте (м)
        """
        x1, y1, x2, y2 = bbox
        
        # 1. Вычисляем размеры bbox в пикселях
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 2. Вычисляем центр bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # 3. РАССТОЯНИЕ ПО ШИРИНЕ
        # Формула: D = (реальная_ширина × фокус) / ширина_в_пикселях
        dist_by_width = (self.PERSON_WIDTH * self.fx) / bbox_width  # в см
        
        # 4. РАССТОЯНИЕ ПО ВЫСОТЕ
        # Формула: D = (реальная_высота × фокус) / высота_в_пикселях
        dist_by_height = (self.PERSON_HEIGHT * self.fy) / bbox_height  # в см
        
        # 5. КОМБИНИРОВАННОЕ РАССТОЯНИЕ (среднее арифметическое)
        
        distance_z_cm = (dist_by_width + dist_by_height) / 2
        distance_z = distance_z_cm / 100  # переводим в метры
        
        # 6. УГОЛ относительно центра камеры
        # Смещение центра bbox от оптического центра камеры
        offset_x_pixels = bbox_center_x - self.cx
        
        # Угол в радианах, затем в градусах
        angle_rad = np.arctan(offset_x_pixels / self.fx)
        angle_deg = np.degrees(angle_rad)
        
        # 7. ГОРИЗОНТАЛЬНОЕ СМЕЩЕНИЕ в метрах
        # Насколько человек смещён влево/вправо от центра
        offset_x = distance_z * np.tan(angle_rad)
        
        # 8. РЕАЛЬНОЕ РАССТОЯНИЕ (с учётом угла)
        # Это гипотенуза треугольника
        distance_real = distance_z / np.cos(angle_rad)
        
        # Возвращаем все результаты
        return {
            'distance_z': distance_z,           # Расстояние вдоль оси камеры (м)
            'distance_real': distance_real,     # Реальное расстояние (м)
            'angle_deg': angle_deg,             # Угол (градусы, + справа, - слева)
            'offset_x': offset_x,               # Смещение влево/вправо (м)
            'dist_by_width': dist_by_width / 100,   # Расстояние по ширине (м)
            'dist_by_height': dist_by_height / 100, # Расстояние по высоте (м)
        }
    
    def detect_people(self, frame, conf_threshold=0.5):
        """
        Детектирует людей на кадре и вычисляет расстояние до них
        
        Args:
            frame: изображение с камеры
            conf_threshold: порог уверенности детекции (0.0 - 1.0)
            
        Returns:
            список детекций, каждая содержит:
                - bbox: [x1, y1, x2, y2]
                - confidence: уверенность детекции
                - distance_z: расстояние вдоль оси (м)
                - distance_real: реальное расстояние (м)
                - angle_deg: угол (градусы)
                - offset_x: смещение (м)
        """
        # Запускаем детекцию YOLO
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        
        # Обрабатываем результаты
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Проверяем, что это человек
                class_id = int(box.cls[0])
                if class_id != self.person_class_id:
                    continue
                
                # Получаем координаты bbox и уверенность
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # Вычисляем расстояние и угол
                measurements = self.calculate_distance_and_angle(bbox)
                
                # Формируем детекцию
                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    **measurements  # Добавляем все измерения
                }
                
                detections.append(detection)
        
        # Сортируем по расстоянию (ближайший первым)
        detections.sort(key=lambda x: x['distance_z'])
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Рисует детекции на кадре с информацией о расстоянии и угле
        
        Args:
            frame: изображение для аннотации
            detections: список детекций
            
        Returns:
            аннотированный кадр
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Рисуем центральную линию (ось камеры)
        cv2.line(annotated, (int(self.cx), 0), (int(self.cx), h), (255, 255, 255), 1)
        cv2.putText(annotated, "CENTER", (int(self.cx) + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Рисуем каждую детекцию
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            distance_z = det['distance_z']
            distance_real = det['distance_real']
            angle = det['angle_deg']
            offset_x = det['offset_x']
            
            # Выбираем цвет в зависимости от расстояния
            if distance_z < 1.5:
                color = (0, 0, 255)      # Красный - очень близко
                warning = "VERY CLOSE!"
            elif distance_z < 3.0:
                color = (0, 165, 255)    # Оранжевый
                warning = "Close"
            elif distance_z < 5.0:
                color = (0, 255, 255)    # Желтый
                warning = ""
            else:
                color = (0, 255, 0)      # Зеленый
                warning = ""
            
            # Рисуем bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем линию от центра к человеку
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            cv2.line(annotated, (int(self.cx), h), (bbox_center_x, bbox_center_y), color, 1)
            
            # Подготавливаем текст с информацией
            texts = [
                f"Person #{i+1} ({confidence:.2f})",
                f"Dist: {distance_z:.2f}m",
                f"Angle: {angle:+.1f}deg",
            ]
            
            if warning:
                texts.append(warning)
            
            # Вычисляем размер фона для текста
            text_height = 0
            max_width = 0
            for text in texts:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_height += th + 5
                max_width = max(max_width, tw)
            
            # Рисуем фон для текста
            cv2.rectangle(annotated, 
                         (x1, y1 - text_height - 10), 
                         (x1 + max_width + 10, y1), 
                         color, -1)
            
            # Рисуем текст
            y_offset = y1 - text_height
            for text in texts:
                y_offset += 15
                cv2.putText(annotated, text, (x1 + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def run_camera(self, camera_id=0):
        """
        Запускает детекцию в реальном времени с камеры
        
        Args:
            camera_id: ID камеры (обычно 0 для веб-камеры)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Ошибка: не могу открыть камеру")
            return
        
        print("\n" + "=" * 70)
        print("ДЕТЕКЦИЯ ЗАПУЩЕНА")
        print("=" * 70)
        print("\nУправление:")
        print("  ESC - выход")
        print("  S   - сохранить текущий кадр")
        print("  P   - пауза/продолжить")
        print("=" * 70 + "\n")
        
        paused = False
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Ошибка чтения кадра")
                    break
                
                # Детектируем людей
                detections = self.detect_people(frame)
                
                # Рисуем аннотации
                annotated_frame = self.draw_detections(frame, detections)
                
                # Вычисляем FPS
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                # Добавляем информацию на экран
                cv2.putText(annotated_frame, f"FPS: {fps_display}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"People: {len(detections)}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Показываем ближайшего человека
                if detections:
                    closest = detections[0]
                    info_text = (f"Closest: {closest['distance_z']:.2f}m "
                               f"@ {closest['angle_deg']:+.1f}deg")
                    cv2.putText(annotated_frame, info_text, (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                display_frame = annotated_frame
            else:
                # В режиме паузы показываем последний кадр
                cv2.putText(display_frame, "PAUSED", 
                           (display_frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Показываем кадр
            cv2.imshow('Person Distance Detection', display_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                # Сохраняем кадр
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"✓ Кадр сохранен: {filename}")
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                print(f"{'⏸ Пауза' if paused else '▶ Продолжение'}")
        
        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("ДЕТЕКЦИЯ ЗАВЕРШЕНА")
        print("=" * 70)


def main():
    """Главная функция"""
    
    # Создаём детектор
    # Убедитесь, что файл camera_calib_result.npz находится в текущей директории
    detector = PersonDistanceDetector(
        calibration_file='camera_calib_result.npz',
        model_path='yolov8n.pt'  # Можно использовать yolov8s.pt, yolov8m.pt для лучшей точности
    )
    
    # Запускаем детекцию с камеры
    detector.run_camera(camera_id=0)


if __name__ == "__main__":
    main()
