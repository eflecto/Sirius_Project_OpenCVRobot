import cv2
import numpy as np

# Цель модуля — выделение силуэта человека и построение «скелета» (тонких линий),
# который подчеркивает структуру формы. Реализованы два подхода:
# 1) Морфологическая скелетизация на основе последовательной эрозии и открытия.
# 2) Тонкая скелетизация (thinning) из ximgproc, если доступна.


def morphological_skeleton(binary_img):
    # Преобразуем бинарное изображение к формату 0/255
    img = (binary_img > 0).astype(np.uint8) * 255
    # Выделяем буфер для скелета
    skel = np.zeros_like(img)
    # Крестовидный структурирующий элемент подчеркивает центральные линии
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        # Эрозия уменьшает объекты, постепенно оголяя их «хребет»
        eroded = cv2.erode(img, element)
        # Открытие (эрозия + дилатация) удаляет тонкие усики, чтобы оставить только основу
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        # Разница между eroded и opened дает тонкие ребра, добавляем их в скелет
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        # Обновляем рабочее изображение, пока все пиксели не исчезнут
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def thinning_skeleton(binary_img):
    try:
        import cv2.ximgproc as ximgproc
        # Алгоритм Zhang-Suen — классический метод тонкой скелетизации
        bin8 = (binary_img > 0).astype(np.uint8) * 255
        skel = ximgproc.thinning(bin8, thinningType=ximgproc.THINNING_ZHANGSUEN)
        return skel
    except Exception:
        # Если модуль ximgproc недоступен, используем морфологическую альтернативу
        return morphological_skeleton(binary_img)


def silhouette_mask(frame,
                    use_hog=False,
                    clahe_clip=2.0,
                    clahe_grid=8,
                    otsu=True,
                    thresh=128,
                    canny_on=True,
                    canny_low=50,
                    canny_high=150,
                    kernel=3,
                    open_iter=1,
                    close_iter=2):
    # Предобработка: перевод в оттенки серого и локальное выравнивание яркости (CLAHE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
    eq = clahe.apply(gray)

    roi = None
    if use_hog:
        # HOG+SVM — классический детектор силуэта человека
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(eq, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) > 0:
            # Выбираем наиболее уверенно детектированный прямоугольник
            idx = int(np.argmax(weights)) if len(weights) == len(rects) else int(np.argmax([w*h for (x,y,w,h) in rects]))
            x, y, w, h = rects[idx]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)
            roi = (x0, y0, x1, y1)

    if roi is not None:
        x0, y0, x1, y1 = roi
        eq_roi = eq[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
        eq_roi = eq

    # Бинаризация: либо автоматический порог Отсу, либо фиксированный порог
    if otsu:
        _, bin_img = cv2.threshold(eq_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bin_img = cv2.threshold(eq_roi, int(thresh), 255, cv2.THRESH_BINARY)

    # Объединение информации пороговой маски и границ (Canny) для более полного силуэта
    edges = cv2.Canny(eq_roi, int(canny_low), int(canny_high)) if canny_on else np.zeros_like(eq_roi)
    mask = cv2.bitwise_or(bin_img, edges)

    # Морфология: открытие очищает шум, закрытие заполняет пробелы внутри силуэта
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel), int(kernel)))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(open_iter))
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(close_iter))

    # Заполнение внешних контуров для получения сплошной фигуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if len(contours) > 0:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Вставляем обработанный ROI обратно в исходные размеры кадра
    out = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    out[y0:y1, x0:x1] = filled
    return out


def skeletonize_person(frame,
                       use_hog=False,
                       method='auto',
                       clahe_clip=2.0,
                       clahe_grid=8,
                       otsu=True,
                       thresh=128,
                       canny_on=True,
                       canny_low=50,
                       canny_high=150,
                       kernel=3,
                       open_iter=1,
                       close_iter=2,
                       color=(0, 0, 255),
                       alpha=0.8):
    # Получаем маску силуэта по конвейеру предварительной обработки
    mask = silhouette_mask(frame, use_hog, clahe_clip, clahe_grid, otsu, thresh,
                           canny_on, canny_low, canny_high, kernel, open_iter, close_iter)

    # Выбор метода скелетизации
    if method == 'auto':
        try:
            skel = thinning_skeleton(mask)
        except Exception:
            skel = morphological_skeleton(mask)
    elif method == 'thin':
        skel = thinning_skeleton(mask)
    else:
        skel = morphological_skeleton(mask)

    # Накладываем цветные «кости» на исходный кадр для наглядности
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[skel > 0] = color
    overlay = cv2.addWeighted(overlay, 1.0, colored, float(alpha), 0)
    return skel, overlay


def _open_camera(index):
    # На Windows пробуем разные бэкенды захвата, чтобы повысить шанс открыть камеру
    for flag in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
        try:
            cam = cv2.VideoCapture(index, flag)
            if cam.isOpened():
                return cam
            cam.release()
        except Exception:
            pass
    cam = cv2.VideoCapture(index)
    return cam if cam.isOpened() else None


if __name__ == "__main__":
    cam = None
    for idx in range(0, 3):
        cam = _open_camera(idx)
        if cam is not None:
            break
    if cam is None:
        print("Не удалось открыть камеру")
        raise SystemExit(1)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Не удалось прочитать кадр")
            break
        skel, overlay = skeletonize_person(frame, use_hog=False, method='auto')
        cv2.imshow('Skeleton', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

