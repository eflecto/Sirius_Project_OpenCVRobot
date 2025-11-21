#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_distance_helper.py

Вспомогательный модуль для расчёта расстояния и угла до человека
по bounding box'ам от YOLOv8 (или любой другой детекции).

Задача модуля:
- принять bbox в пикселях (формат xyxy или xywh),
- учесть калибровку камеры (camera_calib_result.npz),
- учесть реальное разрешение кадра (например, 1280x720),
- оценить расстояние до человека и угол относительно центра камеры,
- выдать координаты цели в системе робота (X вправо, Y вперёд),
- опционально использовать "калибровку по образцу" и сглаживание.

Саму YOLO здесь НЕ используем — только геометрия.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DistanceMeasurement:
    """Результат вычислений."""
    # Геометрия относительно камеры
    distance_z: float          # вдоль оси камеры (м)
    distance_real: float       # гипотенуза (м), с учётом угла
    angle_rad: float           # угол в радианах (по горизонтали)
    angle_deg: float           # угол в градусах

    # Расстояния, посчитанные отдельно по ширине/высоте bbox
    dist_by_width: float       # м
    dist_by_height: float      # м

    # Координаты цели в системе робота (камера в начале, смотрит по оси Y)
    x_robot: float             # вправо + (м)
    y_robot: float             # вперёд + (м)

    # Центр bbox в пикселях
    bbox_center: Tuple[int, int]

    # Оригинальный bbox (x1, y1, x2, y2) в пикселях
    bbox_xyxy: Tuple[float, float, float, float]


class YOLODistanceHelper:
    """
    Класс для расчёта расстояния и угла по bbox'ам от YOLO.

    Типичный сценарий:
        helper = YOLODistanceHelper(
            calibration_file='camera_calib_result.npz',
            default_frame_size=(1280, 720)
        )

        # В цикле по кадрам:
        for frame, yolo_results in ...:
            # Допустим, у YOLO есть bbox человека (x1, y1, x2, y2)
            bbox = (x1, y1, x2, y2)
            meas = helper.from_bbox_xyxy(bbox)

            if meas is not None:
                print(meas.distance_z, meas.angle_deg, meas.x_robot, meas.y_robot)

    По умолчанию:
    - используется модель "человек целиком" с ростом ~1.7 м,
      но её можно заменить калибровкой по образцу (set_reference_distance).
    - включено сглаживание по расстоянию и углу (скользящее окно).
    """

    def __init__(
        self,
        calibration_file: str = 'camera_calib_result.npz',
        default_frame_size: Tuple[int, int] = (1280, 720),
        smooth_window: int = 7,
        person_height_m: float = 1.7,
        person_width_m: float = 0.5,
    ) -> None:
        """
        Args:
            calibration_file: npz с camera_matrix, dist_coeffs, image_size
            default_frame_size: ожидаемое разрешение кадра (w, h), напр. (1280, 720)
            smooth_window: длина окна сглаживания
            person_height_m: предполагаемый рост человека (если нет калибровки по образцу)
            person_width_m: примерная ширина плеч (если нет калибровки по образцу)
        """
        self.default_frame_size = default_frame_size
        self.smooth_window = smooth_window

        # Геометрическая модель человека "по умолчанию"
        self.person_height_m = float(person_height_m)
        self.person_width_m = float(person_width_m)

        # Калибровка камеры
        self.calibration_loaded = False
        self.fx_orig: Optional[float] = None
        self.fy_orig: Optional[float] = None
        self.cx_orig: Optional[float] = None
        self.cy_orig: Optional[float] = None
        self.calib_size: Optional[Tuple[int, int]] = None  # (w, h)

        # Данные для сглаживания (одна цель — один фильтр)
        self._dist_history = []
        self._angle_sin_history = []
        self._angle_cos_history = []

        # Калибровка по образцу
        self.ref_calibrated = False
        self.ref_distance_m: Optional[float] = None
        self.ref_bbox_w: Optional[float] = None
        self.ref_bbox_h: Optional[float] = None

        self._load_calibration(calibration_file)

    # ------------------------------------------------------------------ #
    #                 Загрузка и пересчёт калибровки                     #
    # ------------------------------------------------------------------ #

    def _load_calibration(self, calibration_file: str) -> None:
        """Загрузка параметров камеры из npz."""
        try:
            data = np.load(calibration_file)
            camera_matrix = data['camera_matrix']
            image_size = data.get('image_size', None)

            self.fx_orig = float(camera_matrix[0, 0])
            self.fy_orig = float(camera_matrix[1, 1])
            self.cx_orig = float(camera_matrix[0, 2])
            self.cy_orig = float(camera_matrix[1, 2])

            if image_size is not None:
                # В npz обычно [width, height]
                self.calib_size = (int(image_size[0]), int(image_size[1]))
            else:
                # Если нет, считаем, что калибровка была при 1920x1080
                self.calib_size = (1920, 1080)

            self.calibration_loaded = True

            print("[YOLODistanceHelper] Калибровка загружена из", calibration_file)
            print(f"  fx={self.fx_orig:.2f}, fy={self.fy_orig:.2f}, "
                  f"cx={self.cx_orig:.2f}, cy={self.cy_orig:.2f}")
            print(f"  image_size (calib): {self.calib_size[0]}x{self.calib_size[1]}")
        except Exception as e:
            self.calibration_loaded = False
            self.calib_size = None
            print("[YOLODistanceHelper] Не удалось загрузить калибровку:", e)
            print("  Будет использоваться приближённая модель камеры.")

    def _get_intrinsics_for_frame(self, frame_size: Optional[Tuple[int, int]] = None):
        """
        Пересчитывает (или приближённо оценивает) fx, fy, cx, cy
        под конкретное разрешение кадра.
        """
        if frame_size is None:
            frame_size = self.default_frame_size

        w_frame, h_frame = frame_size

        if self.calibration_loaded and self.calib_size is not None:
            calib_w, calib_h = self.calib_size
            scale_x = w_frame / float(calib_w)
            scale_y = h_frame / float(calib_h)

            fx = self.fx_orig * scale_x
            fy = self.fy_orig * scale_y
            cx = self.cx_orig * scale_x
            cy = self.cy_orig * scale_y
        else:
            # Примитивная модель: фокусное ~ min(w, h), центр — середина кадра
            f = float(min(w_frame, h_frame))
            fx = fy = f
            cx = w_frame / 2.0
            cy = h_frame / 2.0

        return fx, fy, cx, cy

    # ------------------------------------------------------------------ #
    #              Калибровка по образцу (эталонный bbox)               #
    # ------------------------------------------------------------------ #

    def set_reference_distance(self, distance_m: float, bbox_xyxy: Tuple[float, float, float, float]):
        """
        Устанавливает «эталон»: объект стоит на известном расстоянии distance_m,
        при этом у него bbox = bbox_xyxy.

        Далее расстояние будет оцениваться по пропорции:
            D_w = D_ref * (w_ref / w),
            D_h = D_ref * (h_ref / h),
            D   = (D_w + D_h) / 2.
        """
        if distance_m <= 0:
            raise ValueError("distance_m должен быть > 0")

        x1, y1, x2, y2 = bbox_xyxy
        w_box = max(1.0, float(x2 - x1))
        h_box = max(1.0, float(y2 - y1))

        self.ref_distance_m = float(distance_m)
        self.ref_bbox_w = w_box
        self.ref_bbox_h = h_box
        self.ref_calibrated = True

        # сбрасываем сглаживание, чтобы не тащить старые значения
        self._reset_smoothing()

        print("[YOLODistanceHelper] Установлена опорная калибровка:")
        print(f"  D_ref = {self.ref_distance_m:.2f} м,"
              f" w_ref = {self.ref_bbox_w:.1f} px,"
              f" h_ref = {self.ref_bbox_h:.1f} px")

    def reset_reference_distance(self):
        """Сброс опорной калибровки."""
        self.ref_calibrated = False
        self.ref_distance_m = None
        self.ref_bbox_w = None
        self.ref_bbox_h = None
        self._reset_smoothing()
        print("[YOLODistanceHelper] Опорная калибровка сброшена.")

    # ------------------------------------------------------------------ #
    #                       Сглаживание измерений                        #
    # ------------------------------------------------------------------ #

    def _reset_smoothing(self):
        self._dist_history = []
        self._angle_sin_history = []
        self._angle_cos_history = []

    def _smooth(self, distance_z: float, angle_rad: float) -> Tuple[float, float]:
        """
        Простое сглаживание по скользящему окну:
        - расстояние усредняем напрямую,
        - угол — по sin/cos, чтобы не было проблем с переходами.
        """
        # расстояние
        self._dist_history.append(distance_z)
        if len(self._dist_history) > self.smooth_window:
            self._dist_history.pop(0)
        dist_mean = float(sum(self._dist_history) / len(self._dist_history))

        # угол
        self._angle_sin_history.append(np.sin(angle_rad))
        self._angle_cos_history.append(np.cos(angle_rad))
        if len(self._angle_sin_history) > self.smooth_window:
            self._angle_sin_history.pop(0)
            self._angle_cos_history.pop(0)

        sin_mean = float(sum(self._angle_sin_history) / len(self._angle_sin_history))
        cos_mean = float(sum(self._angle_cos_history) / len(self._angle_cos_history))
        angle_smooth = float(np.arctan2(sin_mean, cos_mean))

        return dist_mean, angle_smooth

    # ------------------------------------------------------------------ #
    #                     Основная геометрия по bbox                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_xyxy(bbox, fmt: str = "xyxy") -> Tuple[float, float, float, float]:
        """
        Приводит bbox к формату (x1, y1, x2, y2).

        fmt:
            - "xyxy": (x1, y1, x2, y2)
            - "xywh": (x, y, w, h) — левый верхний угол + ширина/высота
        """
        if fmt == "xyxy":
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2), float(y2)
        elif fmt == "xywh":
            x, y, w, h = bbox
            return float(x), float(y), float(x + w), float(y + h)
        else:
            raise ValueError("fmt должен быть 'xyxy' или 'xywh'")

    def from_bbox_xyxy(
        self,
        bbox_xyxy,
        frame_size: Optional[Tuple[int, int]] = None,
        smooth: bool = True,
    ) -> Optional[DistanceMeasurement]:
        """
        Высчитывает расстояние и угол по bbox (формат x1, y1, x2, y2).

        Args:
            bbox_xyxy: (x1, y1, x2, y2) в пикселях
            frame_size: (width, height) кадра. Если None, берём default_frame_size.
            smooth: включить сглаживание по скользящему окну.

        Returns:
            DistanceMeasurement или None, если bbox некорректен.
        """
        x1, y1, x2, y2 = self._ensure_xyxy(bbox_xyxy, fmt="xyxy")
        if x2 <= x1 or y2 <= y1:
            return None

        if frame_size is None:
            frame_size = self.default_frame_size

        w_frame, h_frame = frame_size
        fx, fy, cx, cy = self._get_intrinsics_for_frame(frame_size)

        # Параметры bbox
        w_box = float(x2 - x1)
        h_box = float(y2 - y1)
        bbox_center_x = x1 + w_box / 2.0
        bbox_center_y = y1 + h_box / 2.0

        # --------------------- Расстояние --------------------- #
        if self.ref_calibrated and \
           self.ref_distance_m is not None and \
           self.ref_bbox_w is not None and \
           self.ref_bbox_h is not None:
            # Калибровка по образцу (пропорции по w и h)
            dist_by_width = self.ref_distance_m * (self.ref_bbox_w / w_box)
            dist_by_height = self.ref_distance_m * (self.ref_bbox_h / h_box)
            distance_z = (dist_by_width + dist_by_height) / 2.0
        else:
            # Модель "человек целиком", рост height_m, плюс ширина
            height_cm = self.person_height_m * 100.0
            width_cm = self.person_width_m * 100.0

            # Расстояние по высоте bbox
            dist_by_height_cm = (height_cm * fy) / max(1.0, h_box)
            # Расстояние по ширине bbox
            dist_by_width_cm = (width_cm * fx) / max(1.0, w_box)

            dist_by_height = dist_by_height_cm / 100.0
            dist_by_width = dist_by_width_cm / 100.0

            distance_z = (dist_by_width + dist_by_height) / 2.0

        # Если калибровка по образцу — dist_by_width/height тоже считаем в метрах:
        if self.ref_calibrated:
            dist_by_width = self.ref_distance_m * (self.ref_bbox_w / w_box)
            dist_by_height = self.ref_distance_m * (self.ref_bbox_h / h_box)

        # ---------------------- Угол -------------------------- #
        offset_x_pixels = bbox_center_x - cx
        angle_rad = float(np.arctan(offset_x_pixels / fx))
        angle_deg = float(np.degrees(angle_rad))

        # ----------------- Сглаживание ------------------------ #
        if smooth:
            distance_z, angle_rad = self._smooth(distance_z, angle_rad)
            angle_deg = float(np.degrees(angle_rad))

        # ------------- Действительное расстояние -------------- #
        cos_a = float(np.cos(angle_rad))
        if abs(cos_a) < 1e-6:
            distance_real = distance_z
        else:
            distance_real = distance_z / cos_a

        # ----------------- Координаты робота ------------------ #
        x_robot = distance_z * float(np.sin(angle_rad))
        y_robot = distance_z * float(np.cos(angle_rad))

        return DistanceMeasurement(
            distance_z=distance_z,
            distance_real=distance_real,
            angle_rad=angle_rad,
            angle_deg=angle_deg,
            dist_by_width=float(dist_by_width),
            dist_by_height=float(dist_by_height),
            x_robot=float(x_robot),
            y_robot=float(y_robot),
            bbox_center=(int(round(bbox_center_x)), int(round(bbox_center_y))),
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        )

    def from_bbox_xywh(
        self,
        bbox_xywh,
        frame_size: Optional[Tuple[int, int]] = None,
        smooth: bool = True,
    ) -> Optional[DistanceMeasurement]:
        """
        То же самое, но bbox задан как (x, y, w, h),
        где (x, y) — левый верхний угол, w, h — ширина и высота.
        """
        x1, y1, x2, y2 = self._ensure_xyxy(bbox_xywh, fmt="xywh")
        return self.from_bbox_xyxy((x1, y1, x2, y2), frame_size=frame_size, smooth=smooth)


# Ниже небольшой пример интеграции с YOLOv8 (только для пояснения, можно удалить):

if __name__ == "__main__":
    """
    Простейший пример использования с результатами YOLOv8:

        results = model(frame)  # где frame — np.ndarray (BGR), размер 1280x720
        bboxes = results[0].boxes.xyxy.cpu().numpy()   # [N, 4]
        classes = results[0].boxes.cls.cpu().numpy()   # [N]
        confid = results[0].boxes.conf.cpu().numpy()   # [N]

        helper = YOLODistanceHelper(
            calibration_file='camera_calib_result.npz',
            default_frame_size=(1280, 720)
        )

        for bbox, cls, conf in zip(bboxes, classes, confid):
            # Фильтруем по классу "person" (0 для COCO)
            if int(cls) != 0:
                continue

            meas = helper.from_bbox_xyxy(bbox)
            if meas is None:
                continue

            print(f"Person: dist={meas.distance_z:.2f} m, angle={meas.angle_deg:.1f} deg,"
                  f" x={meas.x_robot:.2f} m, y={meas.y_robot:.2f} m")
    """
    print("Этот модуль предназначен для импорта. См. пример в docstring __main__.")
