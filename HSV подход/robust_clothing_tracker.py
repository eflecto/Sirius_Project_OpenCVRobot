#!/usr/bin/env python3
"""
Устойчивый трекер одежды с автоматической адаптацией диапазонов

Основная идея:
- Захват эталонного цвета одежды в HSV-пространстве из центрального ROI.
- Адаптивная оценка разброса цвета по ROI и автоматическая настройка диапазонов H/S/V.
- Формирование бинарной маски по цвету с учетом цикличности компоненты Hue.
- Морфологическая очистка маски и выделение наиболее вероятного контура человека.
- Расчет геометрии: центр, оценка расстояния по ширине bbox, угол относительно оси камеры.
- Поддержание траектории и визуализация статуса трекинга, в том числе при кратковременной потере.

Ключевые алгоритмы:
- Использование гистограммы Hue для выбора доминирующего тона в ROI.
- Комбинация среднего и медианного значений для устойчивой оценки S/V.
- Адаптация порогов по стандартному отклонению (robust к освещению и шуму).
- Специальная обработка диапазона Hue около нулевой/максимальной границы (красные оттенки).
"""

import cv2
import numpy as np
import time

class RobustClothingTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Параметры цвета
        # target_hsv — эталонный цвет одежды, на основе которого строится маска.
        # color_history — буфер для потенциального сглаживания/адаптации цвета во времени.
        self.target_hsv = None
        self.color_history = []
        self.max_history = 10
        
        # Адаптивные диапазоны (начинаем с широких)
        # Эти диапазоны будут автоматически корректироваться после захвата цвета
        # в зависимости от дисперсии цвета внутри ROI.
        self.h_range = 25
        self.s_range = 80
        self.v_range = 80
        
        # Состояние трекинга
        # tracking_active — включение логики выделения и сопровождения цели.
        # lost_frames — счетчик последовательных кадров без успешного обнаружения.
        self.tracking_active = False
        self.lost_frames = 0
        self.max_lost_frames = 15
        
        # Траектория
        # Хранит последовательность центров объекта для визуализации движения.
        self.trajectory = []
        self.max_trajectory = 30
        
        # Калибровка камеры (примерные значения)
        # focal_length — эффективное фокусное расстояние в пикселях.
        # person_width_cm — априорная реальная ширина торса человека.
        self.focal_length = 500
        self.person_width_cm = 44  # средняя ширина торса
        
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
        # Чем больше ROI, тем точнее статистика, но выше риск захватить фон.
        roi_size = min(150, min(h, w) // 3)
        cx, cy = w // 2, h // 2
        
        # Вырезаем ROI
        # Используем квадратную область вокруг центра кадра как наиболее
        # вероятное положение цели при захвате цвета.
        roi = frame[cy-roi_size//2:cy+roi_size//2, 
                   cx-roi_size//2:cx+roi_size//2]
        
        if roi.size == 0:
            return False
        
        # Конвертируем в HSV 
        # HSV устойчивее к изменению освещения, чем RGB, и удобен для пороговой фильтрации.
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Метод 1: Средний цвет
        # Усреднение по ROI дает грубую оценку тона/насыщенности/яркости.
        mean_hsv = cv2.mean(roi_hsv)[:3]
        
        # Метод 2: Доминирующий цвет через гистограмму
        # Находим пик в гистограмме Hue — это снижает влияние шумов и фона.
        hist_h = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        dominant_h = np.argmax(hist_h)
        
        # Для S и V берем медиану
        # Медиана менее чувствительна к выбросам (бликам, теням).
        median_s = np.median(roi_hsv[:,:,1])
        median_v = np.median(roi_hsv[:,:,2])
        
        # Комбинируем методы
        # Hue берем по максимуму гистограммы; S,V — среднее между mean и median.
        self.target_hsv = np.array([
            dominant_h,  # Используем доминирующий Hue
            int((mean_hsv[1] + median_s) / 2),  # Среднее между mean и median для S
            int((mean_hsv[2] + median_v) / 2)   # Среднее между mean и median для V
        ], dtype=np.uint8)
        
        # Адаптивно настраиваем диапазоны на основе разброса в ROI
        # Чем выше стандартное отклонение, тем шире пороги, чтобы не потерять цель.
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
        # Используем clip для предотвращения выхода за пределы допустимых диапазонов HSV.
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
        # Hue лежит на окружности [0,179]; красные оттенки могут пересекать границы 0/179.
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
        # Визуальное подсказки для пользователя, где поместить одежду при захвате.
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
        # Легкое размытие снижает шум и стабилизирует сегментацию по цвету.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Небольшое размытие для устойчивости
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Создание маски
        mask = self.create_mask(hsv)
        if mask is None:
            return display, None
        
        # Морфологические операции
        # OPEN удаляет мелкие шумы; CLOSE закрывает отверстия; medianBlur сглаживает края.
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Удаление шума
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # Заполнение дыр
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        # Финальное сглаживание
        mask = cv2.medianBlur(mask, 5)
        
        # Поиск контуров
        # Ищем внешние контуры — они описывают области, соответствующие цвету одежды.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_found = False
        best_contour = None
        
        if contours:
            # Фильтрация по площади
            # Исключаем слишком маленькие объекты и ограничиваем по соотношению сторон.
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
            # Используем моменты для устойчивого определения центра контура.
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w_box // 2
                cy = y + h_box // 2
            
            # Добавляем в траекторию
            # Ограничиваем длину, чтобы визуализация оставалась читаемой.
            self.trajectory.append((cx, cy))
            if len(self.trajectory) > self.max_trajectory:
                self.trajectory.pop(0)
            
            # Оценка расстояния
            # Пропорционально обратной ширине bbox: чем шире, тем ближе объект.
            distance_m = (self.person_width_cm * self.focal_length) / (w_box * 100) if w_box > 0 else 0
            
            # Угол
            # Горизонтальный отклонение центра контура от центра кадра.
            angle_rad = np.arctan((cx - w/2) / self.focal_length)
            angle_deg = np.degrees(angle_rad)
            
            # Координаты робота
            # Преобразуем полярные параметры (distance, angle) в декартовы координаты в системе робота.
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
            # Толщина линии увеличивается по мере давности точки, создавая визуальный градиент.
            for i in range(1, len(self.trajectory)):
                thickness = int(np.sqrt(i / len(self.trajectory)) * 4) + 1
                cv2.line(display, self.trajectory[i-1], self.trajectory[i],
                        (255, 200, 0), thickness)
            
            # Информация
            # Полупрозрачный блок с основными метриками.
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
            # В течение ограниченного окна показываем последнюю известную позицию.
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
        # Блок в правом нижнем углу показывает текущую бинарную маску в уменьшенном виде.
        mask_small = cv2.resize(mask, (w//4, h//4))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_colored[:,:,1] = mask_small  # Зеленый канал
        display[h-h//4-10:h-10, w-w//4-10:w-10] = mask_colored
        
        return display, mask
    
    def adjust_sensitivity(self, increase=True):
        """Регулировка чувствительности"""
        factor = 1.2 if increase else 0.8
        # Мультипликативное изменение порогов. Чем шире диапазоны, тем больше охват,
        # но выше риск ложных срабатываний.
        
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
            
            # Подсказка управления
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
