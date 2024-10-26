MÔ HÌNH CNN TRONG XÁC ĐỊNH LỚP VẬT THỂ TRONG ẢNH

Ứng dụng này gồm các phần chính sau:

- Tải mô hình CNN đã huấn luyện từ file: Mô hình được lưu trong file CNN_ImageProcessing_Manab.h5.
- Dự đoán nhãn cho ảnh: Mô hình sẽ dự đoán lớp của ảnh dựa trên các lớp đã học.
- Gắn nhãn lại: Nếu dự đoán không chính xác, người dùng có thể gắn nhãn thủ công.
- Lưu ảnh đã gắn nhãn lại: Ảnh sẽ được lưu vào thư mục newly_labeled_images để sử dụng trong quá trình huấn luyện lại.
- Huấn luyện lại mô hình: Mô hình sẽ được huấn luyện lại với ảnh mới được gắn nhãn.
