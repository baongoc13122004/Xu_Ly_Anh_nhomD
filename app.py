import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Tải mô hình đã lưu
model = load_model('CNN_ImageProcessing_Manab.h5')

# Danh sách tên các lớp (cập nhật theo mô hình của bạn)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Hàm tiền xử lý ảnh được tải lên
def preprocess_image(image):
    image = image.resize((32, 32))  # Thay đổi kích thước ảnh để phù hợp với mô hình
    image = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch cho ảnh
    return image


# Hàm dự đoán lớp của ảnh và trả về tên lớp
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class], predicted_class  # Trả về tên lớp và chỉ số lớp


# Hàm lưu ảnh và nhãn đúng để huấn luyện lại
def save_image(image, label):
    # Chuyển đổi lại ảnh về định dạng PIL nếu cần thiết
    image = Image.fromarray(np.uint8(image * 255))

    # Tạo thư mục để lưu ảnh nếu chưa có
    if not os.path.exists('corrected_images'):
        os.makedirs('corrected_images')

    # Lưu ảnh cùng với nhãn trong tên file
    image.save(f'corrected_images/{label}_{np.random.randint(1000)}.png')


# Giao diện Streamlit
st.title("Phân loại ảnh với CNN")
st.write("Tải lên một ảnh để phân loại")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh được tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Dự đoán lớp của ảnh
    st.write("Phân loại...")
    predicted_class_name, predicted_class_idx = predict_image(image)

    # Hiển thị kết quả dự đoán
    st.write(f"Lớp dự đoán: {predicted_class_name}")

    # Hỏi người dùng xem dự đoán có chính xác không
    is_correct = st.radio("Dự đoán có chính xác không?", ("Có", "Không"))

    # Nếu dự đoán sai, cho phép người dùng nhập nhãn đúng
    if is_correct == "Không":
        correct_label = st.selectbox("Chọn lớp đúng", class_names)
        save_button = st.button("Lưu ảnh đã chỉnh sửa")

        if save_button:
            save_image(np.array(image) / 255.0, correct_label)
            st.write(f"Đã lưu ảnh với nhãn: {correct_label}")

# Nút để huấn luyện lại mô hình với ảnh đã chỉnh sửa
if st.button("Huấn luyện lại mô hình với ảnh đã chỉnh sửa"):
    st.write("Đang huấn luyện lại mô hình...")
    # Thêm logic huấn luyện lại mô hình với dữ liệu mới tại đây
