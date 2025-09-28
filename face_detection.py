import cv2

class FaceDetector:
    def __init__(self):
        # Загрузка нескольких классификаторов
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            
            # Обнаружение глаз
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 30)
            
            # Обнаружение улыбок
            smiles = self.smile_cascade.detectMultiScale(face_gray, 1.8, 30)
            
            results.append({
                'face': (x, y, w, h),
                'eyes': eyes,
                'smiles': smiles
            })
        
        return results
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            x, y, w, h = detection['face']
            
            # Рисование прямоугольника вокруг лица
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Рисование глаз
            for (ex, ey, ew, eh) in detection['eyes']:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            # Рисование улыбок
            for (sx, sy, sw, sh) in detection['smiles']:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
            
            # Добавление информации о лице
            cv2.putText(frame, f'Face ({w}x{h})', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

def main_advanced():
    detector = FaceDetector()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return
    
    print("Обнаружение лиц")
    print("Нажмите 'q' для выхода")
    print("Нажмите 's' для сохранения кадра")
    
    screens_count = 0
    
    while True:
        is_read_successfully, frame = video_capture.read()
        
        if not is_read_successfully:
            break
        
        # Обнаружение лиц и особенностей
        detections = detector.detect_features(frame)
        
        # Рисование обнаруженных объектов
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Отображение статистики
        cv2.putText(frame_with_detections, f'Faces: {len(detections)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame_with_detections, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Detection', frame_with_detections)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Сохранение кадра
            cv2.imwrite(f'face_detection_{screens_count}.jpg', frame_with_detections)
            print(f"Кадр сохранен как 'face_detection_{screens_count}.jpg'")
            screens_count += 1
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_advanced()