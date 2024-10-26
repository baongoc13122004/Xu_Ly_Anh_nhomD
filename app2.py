import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import os
import shutil
import random

# Tải mô hình đã lưu
model = load_model('CNN_ImageProcessing_Manab.h5')

# Danh sách tên các lớp
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Hàm tiền xử lý ảnh được tải lên
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Hàm dự đoán lớp của ảnh và trả về tên lớp
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class], predicted_class


# Hàm lưu ảnh và nhãn đúng vào thư mục ảnh mới được gắn nhãn
def save_image(image, label):
    if not os.path.exists('newly_labeled_images'):
        os.makedirs('newly_labeled_images')
    image = Image.fromarray(np.uint8(image * 255))
    image.save(f'newly_labeled_images/{label}_{random.randint(1000, 9999)}.png')


# Hàm huấn luyện lại mô hình chỉ với ảnh mới gắn nhãn
def retrain_model():
    images, labels = [], []
    for filename in os.listdir('newly_labeled_images'):
        label = filename.split('_')[0]
        img = Image.open(os.path.join('newly_labeled_images', filename))
        img = img.resize((32, 32))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(class_names.index(label))

    if images and labels:
        x_train = np.array(images)
        y_train = to_categorical(labels, num_classes=len(class_names))

        # Huấn luyện lại mô hình
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=8)

        # Lưu mô hình sau khi huấn luyện lại
        model.save('CNN_ImageProcessing_Manab_retrained.h5')

        # Di chuyển ảnh sau khi huấn luyện xong
        if not os.path.exists('labeled_images'):
            os.makedirs('labeled_images')
        for filename in os.listdir('newly_labeled_images'):
            shutil.move(os.path.join('newly_labeled_images', filename), 'labeled_images')

        st.write("Huấn luyện lại mô hình hoàn tất!")
    else:
        st.write("Không có ảnh nào để huấn luyện lại.")


# Giao diện Streamlit
st.title("Phân loại ảnh với CNN")
st.write("Tải lên một ảnh để phân loại")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    st.write("Phân loại...")
    predicted_class_name, predicted_class_idx = predict_image(image)
    st.write(f"Lớp dự đoán: {predicted_class_name}")

    # Hỏi người dùng xem dự đoán có chính xác không
    is_correct = st.radio("Dự đoán có chính xác không?", ("Có", "Không"))

    # Nếu dự đoán sai, cho phép người dùng nhập nhãn đúng và lưu ảnh nếu nhãn được chỉnh sửa
    if is_correct == "Không":
        correct_label = st.selectbox("Chọn lớp đúng", class_names)
        save_button = st.button("Lưu ảnh đã chỉnh sửa")

        if save_button:
            save_image(np.array(image) / 255.0, correct_label)
            st.write(f"Đã lưu ảnh với nhãn: {correct_label}")

# Nút để huấn luyện lại mô hình với ảnh đã chỉnh sửa
if st.button("Huấn luyện lại mô hình với ảnh đã chỉnh sửa"):
    st.write("Đang huấn luyện lại mô hình...")
    retrain_model()
