# File: chay_tren_pi.py
import cv2
import numpy as np
import time
# Trên Pi 3 không cài tensorflow, chỉ cài tflite_runtime cho nhẹ
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Phòng trường hợp bạn test trên Laptop thì vẫn chạy được bằng tensorflow
    import tensorflow.lite as tflite

# --- CẤU HÌNH ---
MODEL_PATH = "saurang_pi_final.tflite" # Tên file model bạn vừa tải về
IMG_SIZE = 224      # Kích thước ảnh lúc train (BẮT BUỘC KHỚP)
CONFIDENCE = 0.5    # Độ nhạy (0.5 là trung bình)

def main():
    print(f"🔄 Dang load model: {MODEL_PATH}...")
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print("❌ LỖI: Không tìm thấy file model. Hãy chắc chắn file .tflite nằm chung thư mục!")
        return

    # Lấy thông tin Input/Output của model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("📷 Dang khoi dong Camera...")
    cap = cv2.VideoCapture(0) # Số 0 là camera mặc định
    
    # Cài đặt kích thước khung hình camera thấp xuống để giảm tải cho Pi 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_frame_time = 0
    new_frame_time = 0

    print("✅ Bat dau soi rang! Nhan 'q' de thoat.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lỗi Camera!")
            break

        # 1. PRE-PROCESSING (Xử lý ảnh trước khi đưa vào AI)
        # Resize về 224x224
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # Chuẩn hóa màu sắc (chia 255) và đổi sang float32
        input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

        # 2. RUN MODEL (Chạy AI)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start_time = time.time()
        interpreter.invoke() # Đây là lệnh bắt AI suy nghĩ
        
        # 3. POST-PROCESSING (Xử lý kết quả đầu ra)
        # Kết quả là một cái ảnh Mask (đen trắng) bị nén nhỏ
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Xử lý mask: Chỗ nào > 0.5 thì cho là sâu răng
        mask = (output_data > CONFIDENCE).astype(np.uint8) * 255
        
        # Resize mask to bằng kích thước khung hình camera thật để vẽ đè lên
        mask_overlay = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # --- TẠO HIỆU ỨNG TÔ MÀU ---
        # Tìm các đường viền của vùng sâu răng để vẽ cho đẹp
        contours, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vẽ viền màu Đỏ (BGR: 0, 0, 255) lên ảnh gốc
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
        
        # Tô màu bán trong suốt (Overlay)
        # Tạo một lớp màu đỏ
        colored_layer = np.zeros_like(frame)
        colored_layer[:, :, 2] = mask_overlay # Kênh đỏ
        
        # Trộn ảnh gốc và lớp màu đỏ
        frame = cv2.addWeighted(frame, 1.0, colored_layer, 0.4, 0) # 0.4 là độ đậm

        # Hiện FPS lên màn hình
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SO DUNG CUA BAN - BAM 'Q' DE THOAT", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Phat Hien Sau Rang (Pi 3B+)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
