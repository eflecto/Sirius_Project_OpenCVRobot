#!/usr/bin/env python3
"""
Устойчивый трекер одежды с автоматической адаптацией диапазонов
и расчётом расстояния как в yolo_distance_basic, но без YOLO:
- детектируем человека по цвету одежды (торс),
- берем bounding box торса (ширина = ширина торса, высота = талия-плечи),
- считаем расстояние по ширине и высоте bbox через калиброванную камеру.
"""

import cv2
import numpy as np
import time


class RobustClothingTracker:
    def __init__(self, calibration_file='camera_calib_result.npz'):
        self.cap = cv2.VideoCapture(1)

        # Параметры цвета
        self.target_hsv = None
        self.color_history = []
        self.max_history = 10

        # Адаптивные диапазоны HSV
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

        # ---- ПАРАМЕТРЫ ДЛЯ РАССТОЯНИЯ ----
        # Это были "примерные" параметры, оставим как запасной вариант
        self.focal_length = 500  # fallback, если калибровка не загрузится

        # Геометрия торса (НЕ весь рост):
        # ширина = ширина торса по кадру,
        # высота = расстояние от талии до плеч.
        self.TORSO_WIDTH_CM = 44   # средняя ширина торса (можешь подправить под себя)
        self.TORSO_HEIGHT_CM = 70  # примерная высота торса (талия-плечи), тоже можно подкрутить

        # Параметры камеры (заполнятся из калибровки)
        self.calibration_loaded = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self._load_calibration(calibration_file)

        print("=" * 50)
        print("РОБАСТНЫЙ ТРЕКЕР ОДЕЖДЫ + КАЛИБРОВАННОЕ РАССТОЯНИЕ")
        print("=" * 50)
        print("Управление:")
        print("ПРОБЕЛ - захват цвета из центра")
        print("'+'/'-' - увеличить/уменьшить чувствительность")
        print("'r' - сброс")
        print("'q' - выход")
        print("=" * 50)

    # ---------------------------------------------------------------------- #
    #                 КАЛИБРОВКА КАМЕРЫ И РАСЧЁТ РАССТОЯНИЯ                  #
    # ---------------------------------------------------------------------- #

    def _load_calibration(self, calibration_file):
        """Загрузка калибровки камеры как в yolo_distance_basic."""
        try:
            calib_data = np.load(calibration_file)
            camera_matrix = calib_data['camera_matrix']
            dist_coeffs = calib_data['dist_coeffs']  # пока не используем

            self.fx = float(camera_matrix[0, 0])
            self.fy = float(camera_matrix[1, 1])
            self.cx = float(camera_matrix[0, 2])
            self.cy = float(camera_matrix[1, 2])

            self.calibration_loaded = True

            print("[КАЛИБРОВКА] Успешно загружена из", calibration_file)
            print(f"   fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        except Exception as e:
            # Фоллбек — работаем как раньше, с приблизительным focal_length,
            # центр кадра будем брать из размера кадра.
            self.calibration_loaded = False
            print("[КАЛИБРОВКА] Не удалось загрузить:", e)
            print("Работаю с приближённым focal_length =", self.focal_length)

    def calculate_distance_and_angle(self, bbox, frame_size):
        """
        Расчёт расстояния и угла по bbox торса, как в yolo_distance_basic,
        но вместо роста человека используем высоту торса (талия-плечи).

        Args:
            bbox: (x, y, w_box, h_box) - boundingRect торса по маске цвета
            frame_size: (w_frame, h_frame) - размер кадра

        Returns:
            dict или None, если bbox некорректен
        """
        x, y, w_box, h_box = bbox
        w_frame, h_frame = frame_size

        if w_box <= 0 or h_box <= 0:
            return None

        # Центр bbox торса
        bbox_center_x = x + w_box / 2.0
        bbox_center_y = y + h_box / 2.0

        # Параметры камеры
        if self.calibration_loaded:
            fx = self.fx
            fy = self.fy
            cx = self.cx
        else:
            # Если калибровка не загружена — используем фоллбек
            fx = float(self.focal_length)
            fy = fx
            cx = w_frame / 2.0

        # --- 1. Расстояние по ширине торса ---
        # D_w = (реальная_ширина_торса * fx) / ширина_в_пикселях
        dist_by_width_cm = (self.TORSO_WIDTH_CM * fx) / w_box

        # --- 2. Расстояние по высоте торса ---
        # D_h = (реальная_высота_торса * fy) / высота_в_пикселях
        dist_by_height_cm = (self.TORSO_HEIGHT_CM * fy) / h_box

        # --- 3. Итоговое расстояние вдоль оси камеры (среднее) ---
        distance_z_cm = (dist_by_width_cm + dist_by_height_cm) / 2.0
        distance_z = distance_z_cm / 100.0  # в метры

        # --- 4. Угол по смещению от оптического центра ---
        offset_x_pixels = bbox_center_x - cx
        angle_rad = np.arctan(offset_x_pixels / fx)
        angle_deg = np.degrees(angle_rad)

        # --- 5. Горизонтальное смещение и реальное расстояние ---
        offset_x_m = distance_z * np.tan(angle_rad)
        distance_real = distance_z / np.cos(angle_rad)

        return {
            "distance_z": distance_z,               # вдоль оси камеры (то, что нужно роботу вперёд)
            "distance_real": distance_real,         # гипотенуза
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "offset_x": offset_x_m,
            "dist_by_width": dist_by_width_cm / 100.0,
            "dist_by_height": dist_by_height_cm / 100.0,
            "bbox_center": (int(bbox_center_x), int(bbox_center_y)),
        }

    # ---------------------------------------------------------------------- #
    #                        ЦВЕТОВОЙ ТРЕКЕР ОДЕЖДЫ                          #
    # ---------------------------------------------------------------------- #

    def capture_color_adaptive(self, frame):
        """Адаптивный захват цвета с анализом гистограммы"""
        h, w = frame.shape[:2]

        roi_size = min(150, min(h, w) // 3)
        cx, cy = w // 2, h // 2

        roi = frame[cy - roi_size // 2:cy + roi_size // 2,
                    cx - roi_size // 2:cx + roi_size // 2]

        if roi.size == 0:
            return False

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Средний цвет
        mean_hsv = cv2.mean(roi_hsv)[:3]

        # Доминирующий тон по гистограмме
        hist_h = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        dominant_h = np.argmax(hist_h)

        # Медиана S и V
        median_s = np.median(roi_hsv[:, :, 1])
        median_v = np.median(roi_hsv[:, :, 2])

        self.target_hsv = np.array([
            dominant_h,
            int((mean_hsv[1] + median_s) / 2),
            int((mean_hsv[2] + median_v) / 2)
        ], dtype=np.uint8)

        # Адаптация диапазонов
        std_h = np.std(roi_hsv[:, :, 0])
        std_s = np.std(roi_hsv[:, :, 1])
        std_v = np.std(roi_hsv[:, :, 2])

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

        # Обработка цикличности Hue
        if self.target_hsv[0] - self.h_range < 0:
            mask1 = cv2.inRange(
                hsv_frame,
                np.array([0, lower[1], lower[2]], dtype=np.uint8),
                upper
            )
            mask2 = cv2.inRange(
                hsv_frame,
                np.array([180 + (int(self.target_hsv[0]) - self.h_range),
                          lower[1], lower[2]], dtype=np.uint8),
                np.array([179, upper[1], upper[2]], dtype=np.uint8)
            )
            mask = cv2.bitwise_or(mask1, mask2)
        elif self.target_hsv[0] + self.h_range > 179:
            mask1 = cv2.inRange(
                hsv_frame,
                lower,
                np.array([179, upper[1], upper[2]], dtype=np.uint8)
            )
            mask2 = cv2.inRange(
                hsv_frame,
                np.array([0, lower[1], lower[2]], dtype=np.uint8),
                np.array([(int(self.target_hsv[0]) + self.h_range) - 180,
                          upper[1], upper[2]], dtype=np.uint8)
            )
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_frame, lower, upper)

        return mask

    def process_frame(self, frame):
        """Обработка одного кадра: трекинг + расчёт расстояния / координат"""
        display = frame.copy()
        h, w = frame.shape[:2]

        # Показать ROI для захвата цвета
        if not self.tracking_active:
            roi_size = min(150, min(h, w) // 3)
            cx, cy = w // 2, h // 2
            cv2.rectangle(display,
                          (cx - roi_size // 2, cy - roi_size // 2),
                          (cx + roi_size // 2, cy + roi_size // 2),
                          (0, 255, 255), 2)
            cv2.putText(display, "Place target here",
                        (cx - 70, cy - roi_size // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return display, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        mask = self.create_mask(hsv)
        if mask is None:
            return display, None

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        target_found = False
        best_contour = None

        if contours:
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    aspect_ratio = h_cnt / w_cnt if w_cnt > 0 else 0
                    if 0.5 < aspect_ratio < 3.0:
                        valid_contours.append((cnt, area))

            if valid_contours:
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour = valid_contours[0][0]
                target_found = True

        if target_found and best_contour is not None:
            self.lost_frames = 0

            x, y, w_box, h_box = cv2.boundingRect(best_contour)

            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx_obj = int(M["m10"] / M["m00"])
                cy_obj = int(M["m01"] / M["m00"])
            else:
                cx_obj = x + w_box // 2
                cy_obj = y + h_box // 2

            # Добавляем в траекторию
            self.trajectory.append((cx_obj, cy_obj))
            if len(self.trajectory) > self.max_trajectory:
                self.trajectory.pop(0)

            # --- НОВЫЙ РАСЧЁТ РАССТОЯНИЯ И УГЛА ---
            measurements = self.calculate_distance_and_angle(
                (x, y, w_box, h_box),
                (w, h)
            )

            if measurements is not None:
                distance_z = measurements["distance_z"]
                distance_real = measurements["distance_real"]
                angle_rad = measurements["angle_rad"]
                angle_deg = measurements["angle_deg"]
                dist_by_width = measurements["dist_by_width"]
                dist_by_height = measurements["dist_by_height"]

                # Координаты цели в системе робота
                x_robot = distance_z * np.sin(angle_rad)
                y_robot = distance_z * np.cos(angle_rad)
            else:
                distance_z = 0.0
                distance_real = 0.0
                angle_rad = 0.0
                angle_deg = 0.0
                x_robot = 0.0
                y_robot = 0.0
                dist_by_width = 0.0
                dist_by_height = 0.0

            # Визуализация
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(display, (x, y),
                          (x + w_box, y + h_box), (255, 0, 0), 2)
            cv2.circle(display, (cx_obj, cy_obj), 5, (0, 0, 255), -1)

            for i in range(1, len(self.trajectory)):
                thickness = int(np.sqrt(i / len(self.trajectory)) * 4) + 1
                cv2.line(display, self.trajectory[i - 1],
                         self.trajectory[i], (255, 200, 0), thickness)

            # Инфо-бокс
            info_bg = np.zeros((140, 380, 3), dtype=np.uint8)
            info_bg[:] = (40, 40, 40)
            display[10:150, 10:390] = cv2.addWeighted(
                display[10:150, 10:390], 0.3, info_bg, 0.7, 0
            )

            texts = [
                f"Dist Z: {distance_z:.2f} m  (real: {distance_real:.2f} m)",
                f"Angle: {angle_deg:.1f} deg",
                f"Robot X: {x_robot:.2f} m, Y: {y_robot:.2f} m",
                f"By W: {dist_by_width:.2f} m, by H: {dist_by_height:.2f} m",
            ]

            for i, text in enumerate(texts):
                cv2.putText(display, text, (20, 35 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display, "TRACKING", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Потеря цели
            self.lost_frames += 1

            if self.lost_frames < self.max_lost_frames and self.trajectory:
                last_pos = self.trajectory[-1]
                cv2.circle(display, last_pos, 15, (0, 165, 255), 2)
                cv2.putText(display, f"LOST ({self.lost_frames})",
                            (last_pos[0] - 40, last_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 2)
            else:
                cv2.putText(display, "TARGET LOST", (w - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                if self.lost_frames > self.max_lost_frames * 2:
                    self.trajectory = []

        # Мини-карта маски
        mask_small = cv2.resize(mask, (w // 4, h // 4))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 1] = mask_small
        display[h - h // 4 - 10:h - 10,
                w - w // 4 - 10:w - 10] = mask_colored

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

            display, mask = self.process_frame(frame)

            cv2.putText(display, "Press SPACE to capture",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            cv2.imshow('Robust Clothing Tracker', display)
            if mask is not None:
                cv2.imshow('Mask', mask)

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
