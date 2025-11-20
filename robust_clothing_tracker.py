#!/usr/bin/env python3
"""
Устойчивый трекер одежды с автоматической адаптацией диапазонов
и расчётом расстояния как в yolo_distance_basic, но без YOLO:
- детектируем человека по цвету одежды (торс),
- берем bounding box торса (ширина = ширина торса, высота = талия-плечи),
- считаем расстояние по ширине и высоте bbox через калиброванную камеру.

Добавлено:
- корректный учёт того, что калибровка сделана при 1920x1080 (или другом размера),
  параметры пересчитываются под текущее разрешение кадра;
- режимы тестирования: вебкамера, видео, фото;
- крутилка дисторсии (для wide-angle): можно менять силу undistort в рантайме;
- "квадрат поиска": после потери цели трекер некоторое время ищет её только
  в области, где она была, и не перескакивает на другой большой контур.
"""

import cv2
import numpy as np
import os


class RobustClothingTracker:
    def __init__(self, calibration_file='camera_calib_result.npz'):
        self.cap = None  # камера инициализируется в run_*()

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
        self.max_lost_frames = 15  # в течение этого времени ищем только в локальном квадрате

        # Траектория
        self.trajectory = []
        self.max_trajectory = 30

        # Запоминаем последний bbox цели
        self.last_bbox = None  # (x, y, w, h)
        self.search_margin_factor = 1.8  # во сколько раз расширяем bbox для квадрата поиска

        # ---- ПАРАМЕТРЫ ДЛЯ РАССТОЯНИЯ ----
        # Фоллбек, если калибровка не загрузится
        self.focal_length = 500

        # Геометрия торса (НЕ весь рост):
        self.TORSO_WIDTH_CM = 44   # ширина торса
        self.TORSO_HEIGHT_CM = 70  # высота торса (талия-плечи)

        # Параметры камеры (из калибровки)
        self.calibration_loaded = False
        self.fx_orig = None
        self.fy_orig = None
        self.cx_orig = None
        self.cy_orig = None
        self.calib_size = None  # (width, height) при калибровке
        self.dist_coeffs = None  # коэффициенты дисторсии

        # ---- UNDISTORT (крутилка дисторсии) ----
        self.undistort_enabled = True
        self.distortion_scale = 1.0   # 0.0 = без коррекции, 1.0 = как в калибровке
        self.distortion_scale_min = 0.0
        self.distortion_scale_max = 1.5

        self.undistort_map1 = None
        self.undistort_map2 = None
        self.intrinsics_for_frame = None  # матрица камеры для текущего размера + undistort
        self.last_frame_size = None

        self._load_calibration(calibration_file)

        print("=" * 50)
        print("РОБАСТНЫЙ ТРЕКЕР ОДЕЖДЫ + КАЛИБРОВАННОЕ РАССТОЯНИЕ")
        print("=" * 50)
        print("Управление:")
        print("ПРОБЕЛ    - захват цвета из центра")
        print("'+'/'-'   - увеличить/уменьшить чувствительность по HSV")
        print("'[' / ']' - уменьшить/увеличить коррекцию дисторсии")
        print("'d'       - включить/выключить undistort")
        print("'r'       - сброс трекера")
        print("'q' или ESC - выход")
        print("=" * 50)

    # ---------------------------------------------------------------------- #
    #                 КАЛИБРОВКА КАМЕРЫ И РАСЧЁТ РАССТОЯНИЯ                  #
    # ---------------------------------------------------------------------- #

    def _load_calibration(self, calibration_file):
        """Загрузка калибровки камеры (fx, fy, cx, cy, dist_coeffs, image_size)."""
        try:
            calib_data = np.load(calibration_file)
            camera_matrix = calib_data['camera_matrix']
            dist_coeffs = calib_data['dist_coeffs']
            image_size = calib_data.get('image_size', None)

            self.fx_orig = float(camera_matrix[0, 0])
            self.fy_orig = float(camera_matrix[1, 1])
            self.cx_orig = float(camera_matrix[0, 2])
            self.cy_orig = float(camera_matrix[1, 2])

            # dist_coeffs приводим к плоскому float-вектору
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).ravel()

            if image_size is not None:
                # В npz он лежит как [width, height]
                self.calib_size = (int(image_size[0]), int(image_size[1]))
            else:
                # Если нет, по умолчанию считаем калибровку под 1920x1080
                self.calib_size = (1920, 1080)

            self.calibration_loaded = True

            print("[КАЛИБРОВКА] Успешно загружена из", calibration_file)
            print(f"   fx={self.fx_orig:.2f}, fy={self.fy_orig:.2f}, "
                  f"cx={self.cx_orig:.2f}, cy={self.cy_orig:.2f}")
            print(f"   image_size (калибровки): {self.calib_size[0]}x{self.calib_size[1]}")
            print(f"   dist_coeffs: {self.dist_coeffs}")

        except Exception as e:
            self.calibration_loaded = False
            self.dist_coeffs = None
            print("[КАЛИБРОВКА] Не удалось загрузить:", e)
            print("Работаю с приближённым focal_length =", self.focal_length)

    def _build_scaled_camera_matrix(self, frame_size):
        """
        Пересчёт матрицы камеры под текущее разрешение кадра.
        Калибровка делалась, например, при 1920x1080 — это считается масштаб 1.0.
        Для других разрешений масштабируем fx, fy, cx, cy.
        """
        w_frame, h_frame = frame_size

        if self.calibration_loaded and self.calib_size is not None:
            calib_w, calib_h = self.calib_size
            scale_x = w_frame / float(calib_w)
            scale_y = h_frame / float(calib_h)

            fx = self.fx_orig * scale_x
            fy = self.fy_orig * scale_y
            cx = self.cx_orig * scale_x
            cy = self.cy_orig * scale_y

            K_scaled = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]], dtype=np.float32)
            return K_scaled
        else:
            # Фоллбек: примитивная матрица камеры
            fx = float(self.focal_length)
            fy = fx
            cx = w_frame / 2.0
            cy = h_frame / 2.0

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)
            return K

    def _update_undistort_maps(self, frame_size):
        """
        Пересчитываем карты undistort при изменении размера кадра или
        коэффициента дисторсии.
        """
        w_frame, h_frame = frame_size
        K_scaled = self._build_scaled_camera_matrix(frame_size)

        if self.calibration_loaded and self.dist_coeffs is not None:
            # Масштабируем коэффициенты дисторсии
            scaled_dist = (self.dist_coeffs * self.distortion_scale).astype(np.float32)
            scaled_dist = scaled_dist.reshape(-1, 1)

            # Не меняем newCameraMatrix, используем ту же матрицу (без дополнительного кропа)
            newCameraMatrix = K_scaled.copy()

            self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                K_scaled, scaled_dist, None, newCameraMatrix, (w_frame, h_frame), cv2.CV_16SC2
            )

            self.intrinsics_for_frame = newCameraMatrix
        else:
            # Если калибровки нет, undistort не имеет смысла
            self.undistort_map1 = None
            self.undistort_map2 = None
            self.intrinsics_for_frame = K_scaled

        self.last_frame_size = frame_size

    def _ensure_undistort_maps(self, frame_size):
        """Ленивая пересборка карт undistort при изменении размера или scale."""
        if self.last_frame_size != frame_size or self.undistort_map1 is None or self.undistort_map2 is None:
            self._update_undistort_maps(frame_size)

    def _get_effective_intrinsics(self, frame_size):
        """
        Возвращает fx, fy, cx, cy для текущего кадра.
        Если undistort включён, используем матрицу после undistort.
        Если выключён — просто масштабированную базовую матрицу.
        """
        if self.undistort_enabled and self.intrinsics_for_frame is not None:
            K = self.intrinsics_for_frame
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            return fx, fy, cx, cy

        # Без undistort — просто масштабируем камеру
        K = self._build_scaled_camera_matrix(frame_size)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        return fx, fy, cx, cy

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

        # Эффективные параметры камеры для текущего кадра
        fx, fy, cx, cy = self._get_effective_intrinsics(frame_size)

        # --- 1. Расстояние по ширине торса ---
        dist_by_width_cm = (self.TORSO_WIDTH_CM * fx) / w_box

        # --- 2. Расстояние по высоте торса ---
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
            "distance_z": distance_z,               # вдоль оси камеры
            "distance_real": distance_real,         # гипотенуза
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "offset_x": offset_x_m,
            "dist_by_width": dist_by_width_cm / 100.0,
            "dist_by_height": dist_by_height_cm / 100.0,
            "bbox_center": (int(bbox_center_x), int(bbox_center_y)),
        }

    # ---------------------------------------------------------------------- #
    #                   ЛОКАЛЬНЫЙ КВАДРАТ ПОИСКА ПРИ ПОТЕРЕ                  #
    # ---------------------------------------------------------------------- #

    def _get_search_rect(self, bbox, frame_size):
        """
        Строит квадрат поиска вокруг последнего bbox:
        центр квадрата = центр bbox,
        сторона = max(w, h) * search_margin_factor.
        """
        x, y, w, h = bbox
        w_frame, h_frame = frame_size

        side = int(max(w, h) * self.search_margin_factor)
        cx = x + w // 2
        cy = y + h // 2

        sx = max(0, cx - side // 2)
        sy = max(0, cy - side // 2)
        ex = min(w_frame, cx + side // 2)
        ey = min(h_frame, cy + side // 2)

        return sx, sy, ex - sx, ey - sy  # (x, y, w, h)

    def _contour_center(self, cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w / 2.0, y + h / 2.0

    def _is_point_in_rect(self, px, py, rect):
        rx, ry, rw, rh = rect
        return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

    # ---------------------------------------------------------------------- #
    #                        ЦВЕТОВОЙ ТРЕКЕР ОДЕЖДЫ                          #
    # ---------------------------------------------------------------------- #

    def capture_color_adaptive(self, frame):
        """Адаптивный захват цвета с анализом гистограммы."""
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
        self.last_bbox = None  # при новом захвате начинаем с нуля

        print(f"Цвет захвачен: HSV{tuple(self.target_hsv)}")
        print(f"Диапазоны: H±{self.h_range}, S±{self.s_range}, V±{self.v_range}")

        return True

    def create_mask(self, hsv_frame):
        """Создание маски по выбранному цвету."""
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
        """Обработка одного кадра: undistort + трекинг + расчёт расстояния/координат."""
        # Сначала — undistort, если включен
        h, w = frame.shape[:2]

        if self.undistort_enabled and self.calibration_loaded and self.dist_coeffs is not None:
            self._ensure_undistort_maps((w, h))
            if self.undistort_map1 is not None and self.undistort_map2 is not None:
                frame = cv2.remap(frame, self.undistort_map1, self.undistort_map2,
                                  interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]

        display = frame.copy()

        # Показать ROI для захвата цвета, если трекинг не активен
        if not self.tracking_active:
            roi_size = min(150, min(h, w) // 3)
            cx, cy = w // 2, h // 2
            cv2.rectangle(display,
                          (cx - roi_size // 2, cy - roi_size // 2),
                          (cx + roi_size // 2, cy + roi_size // 2),
                          (0, 255, 255), 2)
            cv2.putText(display, "Press SPACE to capture color",
                        (cx - 120, cy - roi_size // 2 - 10),
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

        valid_contours = []
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                    aspect_ratio = h_c / w_c if w_c > 0 else 0
                    if 0.5 < aspect_ratio < 3.0:
                        valid_contours.append((cnt, area))

        w_frame, h_frame = w, h

        # --- ЛОКАЛЬНЫЙ ПОИСК В КВАДРАТЕ ВОКРУГ last_bbox ---
        if valid_contours:
            if self.last_bbox is None or self.lost_frames >= self.max_lost_frames:
                # Инициализация или давно потеряли — можно брать глобально самый большой
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour = valid_contours[0][0]
                target_found = True
            else:
                # Есть предыдущий bbox и цель недавно потеряна —
                # ищем только в квадрате вокруг него
                search_rect = self._get_search_rect(self.last_bbox, (w_frame, h_frame))
                roi_contours = []
                for cnt, area in valid_contours:
                    cx_cnt, cy_cnt = self._contour_center(cnt)
                    if self._is_point_in_rect(cx_cnt, cy_cnt, search_rect):
                        roi_contours.append((cnt, area))

                if roi_contours:
                    roi_contours.sort(key=lambda x: x[1], reverse=True)
                    best_contour = roi_contours[0][0]
                    target_found = True

        if target_found and best_contour is not None:
            self.lost_frames = 0

            x, y, w_box, h_box = cv2.boundingRect(best_contour)
            self.last_bbox = (x, y, w_box, h_box)

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

            # --- РАСЧЁТ РАССТОЯНИЯ И УГЛА ---
            measurements = self.calculate_distance_and_angle(
                (x, y, w_box, h_box),
                (w_frame, h_frame)
            )

            if measurements is not None:
                distance_z = measurements["distance_z"]
                distance_real = measurements["distance_real"]
                angle_rad = measurements["angle_rad"]
                angle_deg = measurements["angle_deg"]
                dist_by_width = measurements["dist_by_width"]
                dist_by_height = measurements["dist_by_height"]
            else:
                distance_z = 0.0
                distance_real = 0.0
                angle_rad = 0.0
                angle_deg = 0.0
                dist_by_width = 0.0
                dist_by_height = 0.0

            # Координаты цели в системе робота (X вправо, Y вперёд)
            x_robot = distance_z * np.sin(angle_rad)
            y_robot = distance_z * np.cos(angle_rad)

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
            info_h = 160
            info_w = 440
            if h > info_h + 20 and w > info_w + 20:
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (40, 40, 40)
                roi = display[10:10 + info_h, 10:10 + info_w]
                blended = cv2.addWeighted(roi, 0.3, info_bg, 0.7, 0)
                display[10:10 + info_h, 10:10 + info_w] = blended

                texts = [
                    f"Dist Z: {distance_z:.2f} m  (real: {distance_real:.2f} m)",
                    f"Angle: {angle_deg:.1f} deg",
                    f"Robot X: {x_robot:.2f} m, Y: {y_robot:.2f} m",
                    f"By W: {dist_by_width:.2f} m, by H: {dist_by_height:.2f} m",
                    f"Undistort: {'ON' if self.undistort_enabled else 'OFF'}  "
                    f"scale={self.distortion_scale:.2f}",
                ]

                for i, text in enumerate(texts):
                    cv2.putText(display, text, (20, 35 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display, "TRACKING", (w - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Потеря цели
            self.lost_frames += 1

            if self.last_bbox is not None:
                # Рисуем квадрат поиска вокруг последнего места цели
                search_rect = self._get_search_rect(self.last_bbox, (w_frame, h_frame))
                sx, sy, sw, sh = search_rect
                cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh),
                              (0, 165, 255), 2)
                cv2.putText(display, f"SEARCHING ({self.lost_frames})",
                            (sx, max(20, sy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 2)

            if self.lost_frames < self.max_lost_frames and self.trajectory:
                last_pos = self.trajectory[-1]
                cv2.circle(display, last_pos, 15, (0, 165, 255), 2)
            else:
                cv2.putText(display, "TARGET LOST", (w - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                # Если долго потеряны — очищаем траекторию и eventually last_bbox
                if self.lost_frames > self.max_lost_frames * 2:
                    self.trajectory = []
                    self.last_bbox = None

        # Мини-карта маски
        if mask is not None:
            mask_small = cv2.resize(mask, (w_frame // 4, h_frame // 4))
            mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 1] = mask_small
            display[h_frame - h_frame // 4 - 10:h_frame - 10,
                    w_frame - w_frame // 4 - 10:w_frame - 10] = mask_colored

        return display, mask

    def adjust_sensitivity(self, increase=True):
        """Регулировка чувствительности по HSV."""
        factor = 1.2 if increase else 0.8

        self.h_range = int(np.clip(self.h_range * factor, 5, 40))
        self.s_range = int(np.clip(self.s_range * factor, 20, 120))
        self.v_range = int(np.clip(self.v_range * factor, 20, 120))

        print(f"Новые диапазоны: H±{self.h_range}, S±{self.s_range}, V±{self.v_range}")

    def adjust_distortion_scale(self, increase=True):
        """Регулировка силы коррекции дисторсии."""
        step = 0.1
        if increase:
            self.distortion_scale = min(self.distortion_scale_max,
                                        self.distortion_scale + step)
        else:
            self.distortion_scale = max(self.distortion_scale_min,
                                        self.distortion_scale - step)
        # Форсируем пересчёт карт при следующем кадре
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.last_frame_size = None

        print(f"Distortion scale: {self.distortion_scale:.2f} "
              f"({'ON' if self.undistort_enabled else 'OFF'})")

    # ---------------------------------------------------------------------- #
    #                            РЕЖИМЫ ЗАПУСКА                              #
    # ---------------------------------------------------------------------- #

    def _handle_common_keys(self, key, frame_for_capture=None):
        """
        Обработка клавиш, общая для всех режимов.
        frame_for_capture нужен, чтобы при нажатии ПРОБЕЛа
        захватывать цвет с текущего кадра.
        """
        if key == ord(' '):
            if frame_for_capture is not None:
                self.capture_color_adaptive(frame_for_capture)
        elif key == ord('+') or key == ord('='):
            self.adjust_sensitivity(increase=True)
        elif key == ord('-') or key == ord('_'):
            self.adjust_sensitivity(increase=False)
        elif key == ord('['):
            self.adjust_distortion_scale(increase=False)
        elif key == ord(']'):
            self.adjust_distortion_scale(increase=True)
        elif key == ord('d'):
            self.undistort_enabled = not self.undistort_enabled
            print("Undistort:", "ON" if self.undistort_enabled else "OFF")
        elif key == ord('r'):
            self.tracking_active = False
            self.target_hsv = None
            self.trajectory = []
            self.lost_frames = 0
            self.last_bbox = None
            print("Трекер сброшен")

    def run_webcam(self, cam_index=0):
        """Работа с живой камеры."""
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print(f"Не удалось открыть камеру {cam_index}")
            return

        print("[MODE] Webcam")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка чтения с камеры")
                break

            display, mask = self.process_frame(frame)

            cv2.putText(display, "WEBCAM MODE",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)

            cv2.imshow('Robust Clothing Tracker', display)
            if mask is not None:
                cv2.imshow('Mask', mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            self._handle_common_keys(key, frame_for_capture=frame)

        self.cap.release()
        cv2.destroyAllWindows()

    def test_on_video(self, video_path):
        """Тестирование на видеофайле."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return

        print(f"[MODE] Video file: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Видео закончилось")
                break

            display, mask = self.process_frame(frame)

            cv2.putText(display, "VIDEO MODE",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)

            cv2.imshow('Robust Clothing Tracker', display)
            if mask is not None:
                cv2.imshow('Mask', mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            self._handle_common_keys(key, frame_for_capture=frame)

        cap.release()
        cv2.destroyAllWindows()

    def test_on_image(self, image_path):
        """Тестирование на одиночном изображении (кадр статичный)."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Не удалось прочитать изображение: {image_path}")
            return

        print(f"[MODE] Image file: {image_path} "
              f"({frame.shape[1]}x{frame.shape[0]})")

        while True:
            display, mask = self.process_frame(frame)

            cv2.putText(display, "IMAGE MODE (press SPACE to capture color)",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)

            cv2.imshow('Robust Clothing Tracker', display)
            if mask is not None:
                cv2.imshow('Mask', mask)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27):
                break

            self._handle_common_keys(key, frame_for_capture=frame)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust clothing tracker + distance (color-based)."
    )
    parser.add_argument("--calib", type=str, default="camera_calib_result.npz",
                        help="Путь к файлу калибровки камеры (.npz)")
    parser.add_argument("--video", type=str, default=None,
                        help="Путь к видеофайлу для теста")
    parser.add_argument("--image", type=str, default=None,
                        help="Путь к изображению для теста")
    parser.add_argument("--cam", type=int, default=0,
                        help="Индекс вебкамеры (по умолчанию 0)")

    args = parser.parse_args()

    tracker = RobustClothingTracker(calibration_file=args.calib)

    # Приоритет: image > video > webcam
    if args.image is not None:
        tracker.test_on_image(args.image)
    elif args.video is not None:
        tracker.test_on_video(args.video)
    else:
        tracker.run_webcam(cam_index=args.cam)
