import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os


# Path ke direktori dataset
dataset_dir = './Data TA' 

# Fungsi untuk membaca dan melakukan preprocessing pada gambar wajah
def preprocess_image(image_path):
    # Baca gambar wajah menggunakan OpenCV
    image = cv2.imread(image_path)

    # Ubah ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menghapus noise dengan filter median
    gray = cv2.medianBlur(gray, ksize=3)

    # Normalisasi nilai piksel
    normalized = gray / 255.0

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

    return np.array(images), np.array(labels)

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
train_images = torch.from_numpy(train_images).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels_encoded).long()
test_images = torch.from_numpy(test_images).unsqueeze(1).float()
test_labels = torch.from_numpy(test_labels_encoded).long()

# Definisikan arsitektur model Anda
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_ratio=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = input_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out

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
        self.squeeze_excitation = SqueezeExcitation(internal_channels)

    def forward(self, x):
        primary_conv = self.primary_conv(x)
        cheap_operation = self.cheap_operation(primary_conv)
        cheap_operation = self.squeeze_excitation(cheap_operation)
        return torch.cat((primary_conv, cheap_operation), 1)

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
def predict_face(image):
    # Konversi gambar ke grayscale
    image = image.convert('L')

    transform = transforms.Compose([
        transforms.Resize((70, 80)),
        transforms.ToTensor()  
    ])

    # Preprocess gambar
    image_tensor = transform(image).unsqueeze(1).float()

    # Lakukan prediksi menggunakan model
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    # Ubah indeks prediksi menjadi nama label kelas
    predicted_label = label_encoder.classes_[predicted_idx]

    # Simpan gambar dalam format PNG
    image.save(f'predicted_face_{predicted_label}.png')

    return predicted_label

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
