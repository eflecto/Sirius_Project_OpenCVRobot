Отдельный модуль, который делает **ровно то же самое, что цветовой трекер**, но для **bbox’ов от YOLOv8**:

* берёт bbox в пикселях (xyxy или xywh) от YOLO;
* учитывает калибровку `camera_calib_result.npz`, пересчитывает под **1280×720** (или любое другое разрешение);
* умеет:

  * считать расстояние по геометрии человека (рост/ширина),
  * либо по **калибровке по образцу** (задал расстояние и текущий bbox → дальше всё по пропорциям);
  * сглаживать расстояние и угол по скользящему окну;
  * отдавать `x_robot`, `y_robot`, `angle`.

Никакой YOLO внутри нет — только обработка bbox’ов.

### Как использовать с YOLOv8 (схема)

```python
from yolo_distance_helper import YOLODistanceHelper

helper = YOLODistanceHelper(
    calibration_file='camera_calib_result.npz',
    default_frame_size=(1280, 720),
)

# в цикле:
results = model(frame)  # frame: (720, 1280, 3)
boxes = results[0].boxes

bboxes = boxes.xyxy.cpu().numpy()  # [N,4]
classes = boxes.cls.cpu().numpy()  # [N]
confs  = boxes.conf.cpu().numpy()  # [N]

for bbox, cls, conf in zip(bboxes, classes, confs):
    if int(cls) != 0:  # 0 = person для COCO
        continue

    meas = helper.from_bbox_xyxy(bbox)  # frame_size по умолчанию (1280,720)
    if meas is None:
        continue

    print(
        f"dist={meas.distance_z:.2f} m, angle={meas.angle_deg:.1f} deg, "
        f"x={meas.x_robot:.2f}, y={meas.y_robot:.2f}"
    )
```

Если хочешь калибровку по реальной сцене:
когда человек стоит на известном расстоянии `D_ref`, и у тебя есть его bbox — вызови:

```python
helper.set_reference_distance(D_ref, bbox_xyxy)
```

после этого расстояние считается **по пропорции от этого эталона**.

