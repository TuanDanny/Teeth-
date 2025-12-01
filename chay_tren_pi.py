import cv2
import numpy as np
import time
import os

# --- CẤU HÌNH QUAN TRỌNG ---
# 1. IP của ESP32 khi ở chế độ tự phát Wifi (AP Mode)
# Cổng là 80, đường dẫn là /stream như code Arduino đã nạp
CAMERA_URL = "http://192.168.4.1:80/stream"

# 2. Tên file model .tflite bạn đã tải về
MODEL_PATH = "saurang_pi_final.tflite"

# 3. Kích thước ảnh training (Bắt buộc phải khớp với lúc train trên Colab)
IMG_SIZE = 224

# 4. Độ nhạy (0.5 là trung bình, nếu nhiễu quá thì tăng lên 0.6 hoặc 0.7)
CONFIDENCE = 0.5 

# --- NHẬP THƯ VIỆN AI ---
print("⚙️ Dang nap thu vien AI...")
try:
    # Ưu tiên dùng tflite_runtime (Nhẹ cho Pi)
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        # Nếu đang test trên laptop cài full tensorflow
        import tensorflow.lite as tflite
    except ImportError:
        print("❌ LỖI: Chua cai thu vien AI!")
        print("👉 Hay chay lenh: pip3 install tflite-runtime")
        exit()

def main():
    # 1. LOAD MODEL
    print(f"🔄 Dang load model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Khong tim thay file '{MODEL_PATH}'")
        return

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # Lấy thông số input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. KẾT NỐI CAMERA (Vòng lặp để thử lại nếu mất kết nối)
    while True:
        print(f"📡 Dang ket noi toi ESP32-CAM: {CAMERA_URL}")
        print("⚠️ Luu y: Pi phai dang ket noi Wifi 'NhaKhoa_Raspi' nhe!")
        
        cap = cv2.VideoCapture(CAMERA_URL)

        if not cap.isOpened():
            print("❌ Khong the ket noi Camera! Dang thu lai sau 2 giay...")
            time.sleep(2)
            continue
        
        print("✅ DA KET NOI THANH CONG! Bat dau soi rang...")
        print("ℹ️ Nhan phim 'q' hoac 'Esc' de thoat.")

        # Biến đếm FPS
        prev_time = 0

        while True:
            ret, frame = cap.read()
            
            # Nếu mất tín hiệu hình ảnh
            if not ret:
                print("⚠️ Mat tin hieu tu ESP32! Dang thu ket noi lai...")
                break # Thoát vòng lặp đọc ảnh để quay lại vòng lặp kết nối

            # --- GIAI ĐOẠN XỬ LÝ AI ---
            try:
                # 1. Resize ảnh về chuẩn 224x224
                img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                
                # 2. Chuẩn hóa về 0-1 và định dạng float32
                input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

                # 3. Đưa vào Model
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # 4. Chạy dự đoán (Inference)
                interpreter.invoke()
                
                # 5. Lấy kết quả mask
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # 6. Xử lý Mask (Ngưỡng lọc)
                # output_data là ảnh mờ 224x224. Chỗ nào > 0.5 là sâu răng
                mask = (output_data > CONFIDENCE).astype(np.uint8) * 255
                
                # Resize mask về bằng kích thước khung hình thật của Camera (VGA 640x480)
                mask_overlay = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                # --- HIỂN THỊ KẾT QUẢ ---
                # Cách 1: Vẽ viền đỏ
                contours, _ = cv2.findContours(mask_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2) # Viền đỏ đậm

                # Cách 2: Tô màu đỏ bán trong suốt
                if np.any(mask_overlay): # Chỉ tô nếu phát hiện sâu răng
                    zeros = np.zeros_like(mask_overlay)
                    # Tạo ảnh màu đỏ (BGR: 0, 0, 255)
                    mask_color = cv2.merge([zeros, zeros, mask_overlay])
                    # Trộn ảnh gốc và màu đỏ
                    frame = cv2.addWeighted(frame, 1, mask_color, 0.5, 0)
                    
                    # Hiện chữ cảnh báo
                    cv2.putText(frame, "PHAT HIEN SAU RANG!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Lỗi xử lý ảnh: {e}")

            # Tính và hiện FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("He Thong Soi Rang (Pi + ESP32)", frame)

            # Phím thoát
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: # q hoặc Esc
                cap.release()
                cv2.destroyAllWindows()
                return # Thoát chương trình

        cap.release() # Giải phóng camera để kết nối lại

if __name__ == "__main__":
    main()
