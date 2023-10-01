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

# Definisikan Ghost Module
class GhostModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, ratio=2):
        super(GhostModule, self).__init__()
        internal_channels = int(output_channels / ratio)

        self.primary_conv = nn.Conv2d(input_channels, internal_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size, stride=1, padding=kernel_size//2, groups=internal_channels, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        primary_conv = self.primary_conv(x)
        cheap_operation = self.cheap_operation(primary_conv)
        return torch.cat((primary_conv, cheap_operation), 1)

# Definisikan GhostNet
class GhostNet(nn.Module):
    def __init__(self, num_classes=8):
        super(GhostNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(16, 16, 1, 1)
        self.stage2 = self._make_stage(16, 24, 2, 2)
        self.stage3 = self._make_stage(24, 40, 2, 2)
        self.stage4 = self._make_stage(40, 80, 3, 2)
        self.stage5 = self._make_stage(80, 160, 3, 2)
        self.stage6 = self._make_stage(160, 320, 1, 1)

        self.conv7 = nn.Sequential(
            nn.Conv2d(320, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, input_channels, output_channels, num_blocks, stride):
        layers = []
        layers.append(GhostModule(input_channels, output_channels, kernel_size=3))
        for _ in range(1, num_blocks):
            layers.append(GhostModule(output_channels, output_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.conv7(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# Buat objek model dengan arsitektur yang sama seperti saat melatih model
model = GhostNet()

# Muat state_dict dari file .pt ke objek model
model.load_state_dict(torch.load('GhostNet.pt'))

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