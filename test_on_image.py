"""
Тестовый скрипт для проверки детектора на изображениях
Полезно для отладки и проверки точности
"""

import cv2
import numpy as np
from yolo_distance_basic import PersonDistanceDetector
import sys


def test_on_image(image_path, detector):
    """
    Тестирует детектор на одном изображении
    
    Args:
        image_path: путь к изображению
        detector: экземпляр PersonDistanceDetector
    """
    print("\n" + "=" * 70)
    print(f"ТЕСТИРОВАНИЕ: {image_path}")
    print("=" * 70)
    
    # Загружаем изображение
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Ошибка: не могу загрузить изображение {image_path}")
        return
    
    print(f"✓ Изображение загружено: {frame.shape[1]}x{frame.shape[0]}")
    
    # Детектируем людей
    print("\nДетекция людей...")
    detections = detector.detect_people(frame)
    
    print(f"✓ Найдено людей: {len(detections)}")
    
    # Выводим информацию о каждом человеке
    if detections:
        print("\nРезультаты:")
        print("-" * 70)
        
        for i, det in enumerate(detections, 1):
            print(f"\nЧеловек #{i}:")
            print(f"  Уверенность детекции: {det['confidence']:.2%}")
            print(f"  Bbox: {det['bbox']}")
            print(f"  Размер bbox: {det['bbox'][2]-det['bbox'][0]}x{det['bbox'][3]-det['bbox'][1]} пикселей")
            print(f"\n  Расстояния:")
            print(f"    По ширине:  {det['dist_by_width']:.2f} м")
            print(f"    По высоте:  {det['dist_by_height']:.2f} м")
            print(f"    Среднее:    {det['distance_z']:.2f} м ✅")
            print(f"    Реальное:   {det['distance_real']:.2f} м")
            print(f"\n  Угол и позиция:")
            print(f"    Угол:       {det['angle_deg']:+.1f}°")
            print(f"    Смещение:   {abs(det['offset_x']):.2f} м {'вправо' if det['offset_x'] > 0 else 'влево'}")
            
            # Оценка надёжности
            diff = abs(det['dist_by_width'] - det['dist_by_height'])
            if diff < 0.5:
                reliability = "✅ Высокая (методы согласованы)"
            elif diff < 1.0:
                reliability = "⚠️ Средняя (небольшое расхождение)"
            else:
                reliability = "❌ Низкая (большое расхождение - человек боком?)"
            
            print(f"\n  Надёжность измерения: {reliability}")
            print(f"  Разница методов: {diff:.2f} м")
    
    # Визуализируем результаты
    annotated = detector.draw_detections(frame, detections)
    
    # Показываем изображение
    print("\nНажмите любую клавишу для продолжения...")
    cv2.imshow('Detection Result', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Предлагаем сохранить
    save = input("\nСохранить результат? (y/n): ").strip().lower()
    if save == 'y':
        output_path = image_path.replace('.', '_detected.')
        cv2.imwrite(output_path, annotated)
        print(f"✓ Сохранено: {output_path}")


def main():
    """Главная функция"""
    
    # Проверяем аргументы
    if len(sys.argv) < 2:
        print("Использование: python test_on_image.py <путь_к_изображению>")
        print("\nПример:")
        print("  python test_on_image.py photo.jpg")
        print("  python test_on_image.py /path/to/image.png")
        return
    
    image_path = sys.argv[1]
    
    # Создаём детектор
    print("Инициализация детектора...")
    detector = PersonDistanceDetector(
        calibration_file='camera_calib_result.npz',
        model_path='yolov8n.pt'
    )
    
    # Тестируем
    test_on_image(image_path, detector)
    
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()
