#!/usr/bin/env python3
"""
Устойчивый трекер одежды с автоматической адаптацией диапазонов
и расчётом расстояния как в yolo_distance_basic, но без YOLO:
- детектируем человека по цвету одежды (торс),
- берем bounding box торса (ширина = ширина торса, высота = талия-плечи),
- считаем расстояние по bbox через калиброванную камеру.

Фичи:
- корректный учёт калибровки (1920x1080 или другое) с пересчётом параметров под текущее разрешение;
- режимы: вебкамера, видео, фото;
- крутилка дисторсии для wide-angle ([ ] и 'd');
- "вечная" зона поиска вокруг last_bbox (не перескакиваем на другие контуры);
- сглаживание расстояния и угла по скользящему окну;
- калибровка расстояния по образцу ('c'): задаёшь реальное расстояние до объекта;
- запись видео окна трекера ('v') в mp4;
- ПАМЯТЬ ВНЕШНОСТИ (appearance): HSV-гистограмма торса, чтобы не путать цель с другими.
"""

import cv2
import numpy as np
import os
from datetime import datetime


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
        self.max_lost_frames = 15  # используется только для отображения состояния

        # Траектория
        self.trajectory = []
        self.max_trajectory = 30

        # Последний bbox цели
        self.last_bbox = None  # (x, y, w, h)
        self.search_margin_factor = 1.8  # во сколько раз расширяем bbox для квадрата поиска

        # ---- ПАРАМЕТРЫ ДЛЯ РАССТОЯНИЯ ----
        # Фоллбек, если калибровка не загрузится
        self.focal_length = 500

        # Приблизительная геометрия торса (используется, если НЕТ калибровки по образцу)
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

        # ---- СГЛАЖИВАНИЕ РАССТОЯНИЯ / УГЛА ----
        self.smooth_window = 7
        self.dist_history = []
        self.angle_sin_history = []
        self.angle_cos_history = []

        # ---- КАЛИБРОВКА ПО ОБРАЗЦУ ----
        self.ref_calibrated = False
        self.ref_distance_m = None
        self.ref_bbox_w = None
        self.ref_bbox_h = None

        # ---- ПАМЯТЬ ВНЕШНОСТИ (appearance) ----
        self.appearance_ref_hist = None
        self.appearance_hist_bins = (16, 16)  # HxS
        # Bhattacharyya: 0 = идеальное совпадение, 1 = очень далеко
        self.appearance_max_bhat_dist = 0.5

        # ---- ЗАПИСЬ ВИДЕО ----
        self.record_enabled = False
        self.video_writer = None
        self.record_path = None
        self.record_fps = 30.0

        self._load_calibration(calibration_file)

        print("=" * 50)
        print("РОБАСТНЫЙ ТРЕКЕР ОДЕЖДЫ + КАЛИБРОВАННОЕ РАССТОЯНИЕ")
        print("=" * 50)
        print("Управление:")
        print("ПРОБЕЛ      - захват цвета из центра")
        print("'+' / '-'   - изменить чувствительность по HSV")
        print("'[' / ']'   - изменить коррекцию дисторсии")
        print("'d'         - включить/выключить undistort")
        print("'c'         - калибровка расстояния по текущему bbox")
        print("'v'         - старт/стоп записи видео окна трекера")
        print("'r'         - сброс трекера (цвет, зоны, калибровки, appearance)")
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
        """Пересчёт матрицы камеры под текущее разрешение кадра."""
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
        """Пересчитываем карты undistort при изменении размера кадра или коэффициента дисторсии."""
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
        if (self.last_frame_size != frame_size or
                self.undistort_map1 is None or self.undistort_map2 is None):
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
        Расчёт расстояния и угла по bbox торса.

        Если self.ref_calibrated == True:
            используем эталонное расстояние + эталонный w/h bbox:
            D_w = D_ref * (w_ref / w), D_h = D_ref * (h_ref / h).

        Иначе:
            используем геометрию торса + fx/fy (старый режим).
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

        # --- Расстояние ---
        if self.ref_calibrated and self.ref_distance_m is not None \
                and self.ref_bbox_w is not None and self.ref_bbox_h is not None:
            # Калибровка по образцу (без явного использования fx/fy, только пропорции)
            dist_by_width = self.ref_distance_m * (self.ref_bbox_w / w_box)
            if self.ref_bbox_h > 0:
                dist_by_height = self.ref_distance_m * (self.ref_bbox_h / h_box)
            else:
                dist_by_height = dist_by_width
            distance_z = (dist_by_width + dist_by_height) / 2.0
        else:
            # Стандартная модель: используем предполагаемые физические размеры торса
            dist_by_width_cm = (self.TORSO_WIDTH_CM * fx) / w_box
            dist_by_height_cm = (self.TORSO_HEIGHT_CM * fy) / h_box
            distance_z_cm = (dist_by_width_cm + dist_by_height_cm) / 2.0
            distance_z = distance_z_cm / 100.0
            dist_by_width = dist_by_width_cm / 100.0
            dist_by_height = dist_by_height_cm / 100.0

        # --- Угол по смещению от оптического центра ---
        offset_x_pixels = bbox_center_x - cx
        angle_rad = np.arctan(offset_x_pixels / fx)
        angle_deg = np.degrees(angle_rad)

        # --- Действительное расстояние (гипотенуза) ---
        distance_real = distance_z / np.cos(angle_rad) if np.cos(angle_rad) != 0 else distance_z

        return {
            "distance_z": distance_z,               # вдоль оси камеры
            "distance_real": distance_real,         # гипотенуза
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "dist_by_width": dist_by_width,
            "dist_by_height": dist_by_height,
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
    #          СГЛАЖИВАНИЕ И СБРОС ФИЛЬТРА / ПАМЯТИ ВНЕШНОСТИ                #
    # ---------------------------------------------------------------------- #

    def _reset_smoothing(self):
        self.dist_history = []
        self.angle_sin_history = []
        self.angle_cos_history = []

    def _smooth_measurements(self, distance_z, angle_rad):
        """
        Сглаживание расстояния и угла по скользящему окну.
        Угол усредняется через sin/cos, чтобы не было проблем на границах.
        """
        # расстояние
        self.dist_history.append(distance_z)
        if len(self.dist_history) > self.smooth_window:
            self.dist_history.pop(0)
        mean_dist = float(sum(self.dist_history) / len(self.dist_history))

        # угол
        self.angle_sin_history.append(np.sin(angle_rad))
        self.angle_cos_history.append(np.cos(angle_rad))
        if len(self.angle_sin_history) > self.smooth_window:
            self.angle_sin_history.pop(0)
            self.angle_cos_history.pop(0)
        mean_sin = float(sum(self.angle_sin_history) / len(self.angle_sin_history))
        mean_cos = float(sum(self.angle_cos_history) / len(self.angle_cos_history))
        smooth_angle_rad = float(np.arctan2(mean_sin, mean_cos))

        return mean_dist, smooth_angle_rad

    # ---------------------------------------------------------------------- #
    #               ПАМЯТЬ ВНЕШНОСТИ: ГИСТОГРАММЫ ПО КОНТУРУ                 #
    # ---------------------------------------------------------------------- #

    def _compute_contour_hist(self, hsv_frame, cnt):
        """
        Строит HSV-гистограмму (H,S) для области внутри контура.
        Возвращает (hist, bbox) или (None, None) если что-то не так.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            return None, None

        patch = hsv_frame[y:y + h, x:x + w]
        if patch.size == 0:
            return None, None

        # Маска внутри bbox для этого контура
        cnt_shifted = cnt.copy()
        cnt_shifted[:, :, 0] -= x
        cnt_shifted[:, :, 1] -= y

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_shifted], -1, 255, -1)

        hist = cv2.calcHist(
            [patch],
            [0, 1],            # H и S
            mask,
            [self.appearance_hist_bins[0], self.appearance_hist_bins[1]],
            [0, 180, 0, 256]
        )
        if hist is None:
            return None, None

        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist, (x, y, w, h)

    # ---------------------------------------------------------------------- #
    #                         КАЛИБРОВКА ПО ОБРАЗЦУ                          #
    # ---------------------------------------------------------------------- #

    def calibrate_reference(self):
        """
        Калибровка расстояния по текущему bbox:
        - берём self.last_bbox (если есть и цель сейчас видна),
        - просим в консоли ввести примерное расстояние (м),
        - запоминаем (D_ref, w_ref, h_ref).
        """
        if self.last_bbox is None or self.lost_frames > 0:
            print("[КАЛИБРОВКА ПО ОБРАЗЦУ] Нельзя — цель не найдена в текущем кадре.")
            return

        x, y, w_box, h_box = self.last_bbox
        print("[КАЛИБРОВКА ПО ОБРАЗЦУ]")
        print(f"Текущий bbox: w={w_box}px, h={h_box}px")

        try:
            d_str = input("Введите примерное расстояние до объекта (в метрах): ")
            ref_dist = float(d_str)
            if ref_dist <= 0:
                raise ValueError
        except Exception:
            print("Некорректное расстояние, калибровка отменена.")
            return

        self.ref_distance_m = ref_dist
        self.ref_bbox_w = float(w_box)
        self.ref_bbox_h = float(h_box)
        self.ref_calibrated = True
        self._reset_smoothing()

        print(f"[OK] Калибровка установлена: D_ref={ref_dist:.2f} м, "
              f"w_ref={w_box}px, h_ref={h_box}px")
        print("Теперь расстояние будет считаться относительно этих значений.")

    # ---------------------------------------------------------------------- #
    #                             ЗАПИСЬ ВИДЕО                               #
    # ---------------------------------------------------------------------- #

    def _start_recording(self, frame, fps=30.0, prefix="session"):
        """Создать VideoWriter и начать запись."""
        if frame is None:
            print("[REC] Нет кадра для определения размера — запись не начата.")
            return

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_track_{ts}.mp4"

        writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[REC] Не удалось открыть файл для записи:", filename)
            return

        self.video_writer = writer
        self.record_enabled = True
        self.record_path = filename
        self.record_fps = fps
        print(f"[REC] Запись начата: {filename} (fps={fps:.1f}, size={w}x{h})")

    def _stop_recording(self):
        """Остановить запись и закрыть файл."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[REC] Запись остановлена. Файл сохранён: {self.record_path}")
        self.video_writer = None
        self.record_enabled = False
        self.record_path = None

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
        self.last_bbox = None          # новая цель — новая зона
        self.ref_calibrated = False    # калибровка по старой цели не актуальна
        self.appearance_ref_hist = None  # сбрасываем эталон внешности
        self._reset_smoothing()

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

            # Индикатор записи
            if self.record_enabled:
                cv2.putText(display, "REC",
                            (w - 80, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            # Пишем кадр, если запись включена
            if self.record_enabled and self.video_writer is not None:
                self.video_writer.write(display)

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

        # --- ВЫБОР КОНТУРА С УЧЁТОМ ВЕЧНОЙ ЗОНЫ ПОИСКА И APPEARANCE ---
        if valid_contours:
            if self.last_bbox is None:
                # Первое появление цели — пока нет last_bbox:
                # если нет appearance_hist -> просто самый большой
                if self.appearance_ref_hist is None:
                    valid_contours.sort(key=lambda x: x[1], reverse=True)
                    best_contour = valid_contours[0][0]
                    target_found = True
                else:
                    # Есть эталон внешности (редкий случай) — выбираем по схожести
                    best_dist = None
                    best_cnt = None
                    for cnt, area in valid_contours:
                        hist, _ = self._compute_contour_hist(hsv, cnt)
                        if hist is None:
                            continue
                        dist = cv2.compareHist(self.appearance_ref_hist, hist,
                                               cv2.HISTCMP_BHATTACHARYYA)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_cnt = cnt
                    if best_cnt is not None and best_dist is not None \
                            and best_dist <= self.appearance_max_bhat_dist:
                        best_contour = best_cnt
                        target_found = True
            else:
                # Есть last_bbox: ИЩЕМ ТОЛЬКО В КВАДРАТЕ ВОКРУГ НЕГО.
                search_rect = self._get_search_rect(self.last_bbox, (w_frame, h_frame))
                roi_contours = []
                for cnt, area in valid_contours:
                    cx_cnt, cy_cnt = self._contour_center(cnt)
                    if self._is_point_in_rect(cx_cnt, cy_cnt, search_rect):
                        roi_contours.append((cnt, area))

                if roi_contours:
                    if self.appearance_ref_hist is not None:
                        # Выбор по минимальному Bhattacharyya расстоянию
                        best_dist = None
                        best_cnt = None
                        for cnt, area in roi_contours:
                            hist, _ = self._compute_contour_hist(hsv, cnt)
                            if hist is None:
                                continue
                            dist = cv2.compareHist(self.appearance_ref_hist, hist,
                                                   cv2.HISTCMP_BHATTACHARYYA)
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_cnt = cnt
                        if best_cnt is not None and best_dist is not None \
                                and best_dist <= self.appearance_max_bhat_dist:
                            best_contour = best_cnt
                            target_found = True
                        # иначе: никто не похож достаточно => цель "не найдена", но зону не сбрасываем
                    else:
                        # Эталон внешности ещё не сформирован — выбираем по площади
                        roi_contours.sort(key=lambda x: x[1], reverse=True)
                        best_contour = roi_contours[0][0]
                        target_found = True

        if target_found and best_contour is not None:
            self.lost_frames = 0

            x, y, w_box, h_box = cv2.boundingRect(best_contour)
            self.last_bbox = (x, y, w_box, h_box)

            # Если ещё нет эталонной гистограммы, создаём её
            if self.appearance_ref_hist is None:
                ref_hist, _ = self._compute_contour_hist(hsv, best_contour)
                if ref_hist is not None:
                    self.appearance_ref_hist = ref_hist
                    print("[APPEARANCE] Эталон внешности сохранён.")

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
                raw_distance_z = measurements["distance_z"]
                raw_distance_real = measurements["distance_real"]
                raw_angle_rad = measurements["angle_rad"]
                dist_by_width = measurements["dist_by_width"]
                dist_by_height = measurements["dist_by_height"]

                # Сглаживаем расстояние и угол
                distance_z, angle_rad = self._smooth_measurements(
                    raw_distance_z, raw_angle_rad
                )
                angle_deg = np.degrees(angle_rad)
                # Пересчитываем реальное расстояние по сглаженным значениям
                distance_real = distance_z / np.cos(angle_rad) if np.cos(angle_rad) != 0 else raw_distance_real
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
            info_h = 210
            info_w = 500
            if h > info_h + 20 and w > info_w + 20:
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (40, 40, 40)
                roi_info = display[10:10 + info_h, 10:10 + info_w]
                blended = cv2.addWeighted(roi_info, 0.3, info_bg, 0.7, 0)
                display[10:10 + info_h, 10:10 + info_w] = blended

                if self.ref_calibrated:
                    ref_str = f"D_ref={self.ref_distance_m:.2f}m, " \
                              f"w_ref={self.ref_bbox_w:.0f}px, h_ref={self.ref_bbox_h:.0f}px"
                else:
                    ref_str = "D_ref: not set (using torso model)"

                app_str = "SET" if self.appearance_ref_hist is not None else "NOT SET"

                texts = [
                    f"Dist Z (smoothed): {distance_z:.2f} m  (real: {distance_real:.2f} m)",
                    f"Angle (smoothed):  {angle_deg:.1f} deg",
                    f"Robot X: {x_robot:.2f} m, Y: {y_robot:.2f} m",
                    f"By W: {dist_by_width:.2f} m, by H: {dist_by_height:.2f} m",
                    f"Undistort: {'ON' if self.undistort_enabled else 'OFF'}  "
                    f"scale={self.distortion_scale:.2f}",
                    f"Ref calib: {ref_str}",
                    f"Appearance ref: {app_str}  (max dist={self.appearance_max_bhat_dist:.2f})",
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
                # Траекторию иногда чистим, но last_bbox не трогаем,
                # чтобы зона поиска оставалась.
                if self.lost_frames > self.max_lost_frames * 4:
                    self.trajectory = []

        # Мини-карта маски
        if mask is not None:
            mask_small = cv2.resize(mask, (w_frame // 4, h_frame // 4))
            mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 1] = mask_small
            display[h_frame - h_frame // 4 - 10:h_frame - 10,
                    w_frame - w_frame // 4 - 10:w_frame - 10] = mask_colored

        # Индикатор записи
        if self.record_enabled:
            cv2.putText(display, "REC",
                        (w_frame - 80, h_frame - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        # Запись кадра в файл, если запись включена
        if self.record_enabled and self.video_writer is not None:
            self.video_writer.write(display)

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

    def _handle_common_keys(self, key, frame_for_capture=None,
                            frame_for_record=None, fps=None, prefix="session"):
        """
        Обработка клавиш, общая для всех режимов.
        frame_for_capture нужен, чтобы при нажатии ПРОБЕЛа
        захватывать цвет с текущего кадра.
        frame_for_record + fps + prefix нужны для старта/остановки записи.
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
        elif key == ord('c'):
            # Калибровка расстояния по текущему bbox
            self.calibrate_reference()
        elif key == ord('v'):
            # Старт/стоп записи
            if not self.record_enabled:
                if frame_for_record is not None:
                    use_fps = fps if (fps is not None and fps > 0) else 30.0
                    self._start_recording(frame_for_record, use_fps, prefix)
                else:
                    print("[REC] Нет кадра для записи.")
            else:
                self._stop_recording()
        elif key == ord('r'):
            self.tracking_active = False
            self.target_hsv = None
            self.trajectory = []
            self.lost_frames = 0
            self.last_bbox = None
            self.ref_calibrated = False
            self.ref_distance_m = None
            self.ref_bbox_w = None
            self.ref_bbox_h = None
            self.appearance_ref_hist = None
            self._reset_smoothing()
            print("Трекер сброшен")

    def run_webcam(self, cam_index=0):
        """Работа с живой камеры."""
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print(f"Не удалось открыть камеру {cam_index}")
            return

        # Пытаемся узнать fps камеры
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        print(f"[MODE] Webcam (fps ~ {fps:.1f})")

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

            self._handle_common_keys(
                key,
                frame_for_capture=frame,
                frame_for_record=display,
                fps=fps,
                prefix="webcam"
            )

        if self.record_enabled:
            self._stop_recording()

        self.cap.release()
        cv2.destroyAllWindows()

    def test_on_video(self, video_path):
        """Тестирование на видеофайле."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return

        # fps видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"[MODE] Video file: {video_path} (fps ~ {fps:.1f})")

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

            self._handle_common_keys(
                key,
                frame_for_capture=frame,
                frame_for_record=display,
                fps=fps,
                prefix=base_name
            )

        if self.record_enabled:
            self._stop_recording()

        cap.release()
        cv2.destroyAllWindows()

    def test_on_image(self, image_path):
        """Тестирование на одиночном изображении (статичный кадр)."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Не удалось прочитать изображение: {image_path}")
            return

        print(f"[MODE] Image file: {image_path} "
              f"({frame.shape[1]}x{frame.shape[0]})")

        fps = 30.0  # условный fps для записи

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

            self._handle_common_keys(
                key,
                frame_for_capture=frame,
                frame_for_record=display,
                fps=fps,
                prefix="image"
            )

        if self.record_enabled:
            self._stop_recording()

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
