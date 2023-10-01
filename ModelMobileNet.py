import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os


# Path ke direktori dataset
dataset_dir = './ExtractedFaces' 
# Impor library OpenCV
import cv2

# Load classifier Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Fungsi untuk mendeteksi wajah dalam gambar
def detect_face(image):
    # Konversi gambar ke dalam format OpenCV
    image_cv2 = np.array(image)
    # Ubah ke skala abu-abu
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Jika wajah ditemukan, kembalikan gambar dengan kotak wajah
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return Image.fromarray(image_cv2), True
    else:
        return image, False

def preprocess_image(image_path):
    # Baca gambar wajah menggunakan OpenCV
    image = cv2.imread(image_path)

    # Resize gambar ke ukuran 70x80 pixel
    resized_image = cv2.resize(image, (100, 100))

    # Ubah ke skala abu-abu
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Menghapus noise dengan filter median
    gray = cv2.medianBlur(gray, ksize=3)

    # Normalisasi nilai piksel dan ubah ke tipe data float32
    normalized = gray.astype(np.float32) / 255.0

    return normalized

# Fungsi untuk memuat dataset
def load_dataset():
    images = []
    labels = []

    # Loop melalui setiap direktori kelas dalam dataset
    for class_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, class_dir)):
            class_label = class_dir

            # Loop melalui setiap gambar dalam direktori kelas
            for image_file in os.listdir(os.path.join(dataset_dir, class_dir)):
                if image_file.endswith('.png'):
                    image_path = os.path.join(dataset_dir, class_dir, image_file)

                    # Preprocess gambar wajah
                    preprocessed_image = preprocess_image(image_path)

                    # Tambahkan gambar dan label ke list
                    images.append(preprocessed_image)
                    labels.append(class_label)

    # Mengubah list menjadi numpy arrays dan menampilkan tipe datanya
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Memuat dataset
images, labels = load_dataset()

# Melakukan pembagian dataset menjadi set pelatihan dan pengujian
from sklearn.model_selection import train_test_split

# Membagi dataset menjadi set pelatihan dan pengujian dengan perbandingan 60:40
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.4, train_size=0.6, random_state=42)

# Melakukan one-hot encoding pada label
from sklearn.preprocessing import LabelEncoder

# Membuat objek LabelEncoder
label_encoder = LabelEncoder()

# Melakukan transformasi label menjadi angka
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Konversi numpy array menjadi PyTorch tensors
train_images_face = torch.from_numpy(train_images).unsqueeze(1).float()
train_labels_face = torch.from_numpy(train_labels_encoded).long()
test_images_face = torch.from_numpy(test_images).unsqueeze(1).float()
test_labels_face = torch.from_numpy(test_labels_encoded).long()

from torchvision.models import mobilenet_v2

# Membuat model MobileNet untuk data grayscale
class MobileNetGrayscale(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetGrayscale, self).__init__()
        self.model = mobilenet_v2(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)

        # Menyesuaikan jumlah kelas di layer akhir
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

num_classes=8

# Buat objek model dengan arsitektur yang sama seperti saat melatih model
model = MobileNetGrayscale(num_classes)

# Muat state_dict dari file .pt ke objek model
model.load_state_dict(torch.load('MobileNet.pt'))

# Set model dalam mode evaluasi
model.eval()

# Inisialisasi Flask
app = Flask(__name__)

# Fungsi untuk memproses gambar dan melakukan prediksi
import numpy as np

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_face(image):
    # Deteksi wajah dalam gambar
    detected_image, face_detected = detect_face(image)
    
    if not face_detected:
        return 'No Face Detected'
    
    # Konversi gambar ke grayscale
    gray_image = detected_image.convert('L')

    transform = transforms.Compose([
        transforms.Resize((70, 80)),
        transforms.ToTensor()
    ])

    # Preprocess gambar
    image_tensor = transform(gray_image).unsqueeze(1).float()

    # Lakukan prediksi menggunakan model
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    # Ubah indeks prediksi menjadi nama label kelas
    predicted_label = label_encoder.classes_[predicted_idx]

    # Hitung akurasi prediksi
    confidence = torch.softmax(outputs, dim=1)[0][predicted_idx.item()].item() * 100

    return f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}%'


@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    # Baca gambar dari permintaan POST
    image_file = request.files['image']
    image = Image.open(image_file)

    try:
        # Proses gambar untuk pengenalan wajah
        prediction = predict_face(image)

        # Misalnya, tambahkan pesan status sukses
        if prediction == 'No Face Detected':
            response = {
                'status': 'success',
                'message': 'No face detected in the image.'
            }
        else:
            response = {
                'status': 'success',
                'prediction': prediction,
                'message': 'Face recognized successfully.'
            }

        # Kembalikan respon JSON
        return jsonify(response)

    except Exception as e:
        # Jika terjadi error saat pengenalan wajah, berikan pesan error yang lebih spesifik
        return jsonify({'error': 'Face recognition failed: {}'.format(str(e))})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)