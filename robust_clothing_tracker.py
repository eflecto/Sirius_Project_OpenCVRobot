#!/usr/bin/env python3
"""
Устойчивый трекер одежды с автоматической адаптацией диапазонов
"""

import cv2
import numpy as np
import time

class RobustClothingTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Параметры цвета
        self.target_hsv = None
        self.color_history = []
        self.max_history = 10
        
        # Адаптивные диапазоны (начинаем с широких)
        self.h_range = 25
        self.s_range = 80
        self.v_range = 80
        
        # Состояние трекинга
        self.tracking_active = False
        self.lost_frames = 0
        self.max_lost_frames = 15
        
        # Траектория
        self.trajectory = []
        self.max_trajectory = 30
        
        # Калибровка камеры (примерные значения)
        self.focal_length = 500
        self.person_width_cm = 45  # средняя ширина торса
        
        print("="*50)
        print("РОБАСТНЫЙ ТРЕКЕР ОДЕЖДЫ")
        print("="*50)
        print("Управление:")
        print("ПРОБЕЛ - захват цвета из центра")
        print("'+'/'-' - увеличить/уменьшить чувствительность")
        print("'r' - сброс")
        print("'q' - выход")
        print("="*50)
        
    def capture_color_adaptive(self, frame):
        """Адаптивный захват цвета с анализом гистограммы"""
        h, w = frame.shape[:2]
        
        # Определяем размер ROI
        roi_size = min(150, min(h, w) // 3)
        cx, cy = w // 2, h // 2
        
        # Вырезаем ROI
        roi = frame[cy-roi_size//2:cy+roi_size//2, 
                   cx-roi_size//2:cx+roi_size//2]
        
        if roi.size == 0:
            return False
        
        # Конвертируем в HSV
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Метод 1: Средний цвет
        mean_hsv = cv2.mean(roi_hsv)[:3]
        
        # Метод 2: Доминирующий цвет через гистограмму
        # Находим пик в гистограмме Hue
        hist_h = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        dominant_h = np.argmax(hist_h)
        
        # Для S и V берем медиану
        median_s = np.median(roi_hsv[:,:,1])
        median_v = np.median(roi_hsv[:,:,2])
        
        # Комбинируем методы
        self.target_hsv = np.array([
            dominant_h,  # Используем доминирующий Hue
            int((mean_hsv[1] + median_s) / 2),  # Среднее между mean и median для S
            int((mean_hsv[2] + median_v) / 2)   # Среднее между mean и median для V
        ], dtype=np.uint8)
        
        # Адаптивно настраиваем диапазоны на основе разброса в ROI
        std_h = np.std(roi_hsv[:,:,0])
        std_s = np.std(roi_hsv[:,:,1])
        std_v = np.std(roi_hsv[:,:,2])
        
        self.h_range = min(30, max(10, int(std_h * 1.5)))
        self.s_range = min(100, max(30, int(std_s * 2)))
        self.v_range = min(100, max(30, int(std_v * 2)))
        
        self.tracking_active = True
        self.lost_frames = 0
        
        print(f"Цвет захвачен: HSV{tuple(self.target_hsv)}")
        print(f"Диапазоны: H±{self.h_range}, S±{self.s_range}, V±{self.v_range}")
        
        return True
    
    def create_mask(self, hsv_frame):
        """Создание маски с защитой от ошибок"""
        if self.target_hsv is None:
            return None
        
        # Безопасное создание границ
        lower = np.array([
            np.clip(int(self.target_hsv[0]) - self.h_range, 0, 179),
            np.clip(int(self.target_hsv[1]) - self.s_range, 0, 255),
            np.clip(int(self.target_hsv[2]) - self.v_range, 0, 255)
        ], dtype=np.uint8)
        
        upper = np.array([
            np.clip(int(self.target_hsv[0]) + self.h_range, 0, 179),
            np.clip(int(self.target_hsv[1]) + self.s_range, 0, 255),
            np.clip(int(self.target_hsv[2]) + self.v_range, 0, 255)
        ], dtype=np.uint8)
        
        # Особая обработка для Hue (циклический)
        if self.target_hsv[0] - self.h_range < 0:
            # Красный цвет около 0/180
            mask1 = cv2.inRange(hsv_frame, 
                              np.array([0, lower[1], lower[2]], dtype=np.uint8),
                              upper)
            mask2 = cv2.inRange(hsv_frame,
                              np.array([180 + (int(self.target_hsv[0]) - self.h_range), lower[1], lower[2]], dtype=np.uint8),
                              np.array([179, upper[1], upper[2]], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        elif self.target_hsv[0] + self.h_range > 179:
            # Красный цвет около 180/0
            mask1 = cv2.inRange(hsv_frame,
                              lower,
                              np.array([179, upper[1], upper[2]], dtype=np.uint8))
            mask2 = cv2.inRange(hsv_frame,
                              np.array([0, lower[1], lower[2]], dtype=np.uint8),
                              np.array([(int(self.target_hsv[0]) + self.h_range) - 180, upper[1], upper[2]], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Обычный случай
            mask = cv2.inRange(hsv_frame, lower, upper)
        
        return mask
    
    def process_frame(self, frame):
        """Обработка кадра"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Показываем ROI если не в режиме трекинга
        if not self.tracking_active:
            roi_size = min(150, min(h, w) // 3)
            cx, cy = w // 2, h // 2
            cv2.rectangle(display,
                        (cx-roi_size//2, cy-roi_size//2),
                        (cx+roi_size//2, cy+roi_size//2),
                        (0, 255, 255), 2)
            cv2.putText(display, "Place target here", 
                       (cx-70, cy-roi_size//2-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return display, None
        
        # Преобразование в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Небольшое размытие для устойчивости
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Создание маски
        mask = self.create_mask(hsv)
        if mask is None:
            return display, None
        
        # Морфологические операции
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Удаление шума
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # Заполнение дыр
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        # Финальное сглаживание
        mask = cv2.medianBlur(mask, 5)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_found = False
        best_contour = None
        
        if contours:
            # Фильтрация по площади
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:  # Минимальная площадь
                    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    aspect_ratio = h_cnt / w_cnt if w_cnt > 0 else 0
                    # Фильтр по соотношению сторон (человек обычно выше чем шире)
                    if 0.5 < aspect_ratio < 3.0:
                        valid_contours.append((cnt, area))
            
            if valid_contours:
                # Выбираем самый большой контур
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour = valid_contours[0][0]
                target_found = True
        
        if target_found and best_contour is not None:
            # Сброс счетчика потери
            self.lost_frames = 0
            
            # Bounding box
            x, y, w_box, h_box = cv2.boundingRect(best_contour)
            
            # Центр масс
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w_box // 2
                cy = y + h_box // 2
            
            # Добавляем в траекторию
            self.trajectory.append((cx, cy))
            if len(self.trajectory) > self.max_trajectory:
                self.trajectory.pop(0)
            
            # Оценка расстояния
            distance_m = (self.person_width_cm * self.focal_length) / (w_box * 100) if w_box > 0 else 0
            
            # Угол
            angle_rad = np.arctan((cx - w/2) / self.focal_length)
            angle_deg = np.degrees(angle_rad)
            
            # Координаты робота
            x_robot = distance_m * np.sin(angle_rad)
            y_robot = distance_m * np.cos(angle_rad)
            
            # Визуализация
            # Контур
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)
            
            # Bounding box
            cv2.rectangle(display, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
            
            # Центр
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            
            # Траектория
            for i in range(1, len(self.trajectory)):
                thickness = int(np.sqrt(i / len(self.trajectory)) * 4) + 1
                cv2.line(display, self.trajectory[i-1], self.trajectory[i],
                        (255, 200, 0), thickness)
            
            # Информация
            info_bg = np.zeros((120, 350, 3), dtype=np.uint8)
            info_bg[:] = (40, 40, 40)
            display[10:130, 10:360] = cv2.addWeighted(display[10:130, 10:360], 0.3, info_bg, 0.7, 0)
            
            texts = [
                f"Distance: {distance_m:.2f} m",
                f"Angle: {angle_deg:.1f} deg",
                f"Robot X: {x_robot:.2f} m, Y: {y_robot:.2f} m",
                f"Confidence: {cv2.contourArea(best_contour):.0f} px"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(display, text, (20, 35 + i*25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Статус
            cv2.putText(display, "TRACKING", (w-120, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        else:
            # Цель потеряна
            self.lost_frames += 1
            
            if self.lost_frames < self.max_lost_frames and len(self.trajectory) > 0:
                # Показываем последнюю известную позицию
                last_pos = self.trajectory[-1]
                cv2.circle(display, last_pos, 15, (0, 165, 255), 2)
                cv2.putText(display, f"LOST ({self.lost_frames})", 
                          (last_pos[0]-40, last_pos[1]-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # Полная потеря
                cv2.putText(display, "TARGET LOST", (w-150, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.lost_frames > self.max_lost_frames * 2:
                    self.trajectory = []
        
        # Мини-карта маски
        mask_small = cv2.resize(mask, (w//4, h//4))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_colored[:,:,1] = mask_small  # Зеленый канал
        display[h-h//4-10:h-10, w-w//4-10:w-10] = mask_colored
        
        return display, mask
    
    def adjust_sensitivity(self, increase=True):
        """Регулировка чувствительности"""
        factor = 1.2 if increase else 0.8
        
        self.h_range = int(np.clip(self.h_range * factor, 5, 40))
        self.s_range = int(np.clip(self.s_range * factor, 20, 120))
        self.v_range = int(np.clip(self.v_range * factor, 20, 120))
        
        print(f"Новые диапазоны: H±{self.h_range}, S±{self.s_range}, V±{self.v_range}")
    
    def run(self):
        """Основной цикл"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка чтения с камеры")
                break
            
            # Обработка кадра
            display, mask = self.process_frame(frame)
            
            # Показ FPS
            cv2.putText(display, f"Press SPACE to capture", 
                      (10, display.shape[0]-20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Robust Clothing Tracker', display)
            if mask is not None:
                cv2.imshow('Mask', mask)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self.capture_color_adaptive(frame)
            elif key == ord('+') or key == ord('='):
                self.adjust_sensitivity(increase=True)
            elif key == ord('-') or key == ord('_'):
                self.adjust_sensitivity(increase=False)
            elif key == ord('r'):
                self.tracking_active = False
                self.target_hsv = None
                self.trajectory = []
                print("Трекер сброшен")
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = RobustClothingTracker()
    tracker.run()
