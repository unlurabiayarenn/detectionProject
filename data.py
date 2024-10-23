import os
import cv2
import numpy as np

def load_data(data_dir):
    images = []
    labels = []
    for label in ['closed', 'open']:
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Görselleri yeniden boyutlandırın
            images.append(img)
            labels.append(0 if label == 'closed' else 1)  # Kapalı için 0, açık için 1
    return np.array(images), np.array(labels)

# Veri seti dizinini tanımlayın
train_dir = 'train'
test_dir = 'test'

# Eğitim verisini yükleyin
X_train, y_train = load_data(train_dir)

#test verisinin yüklenmesi
X_test, y_test = load_data(test_dir) if os.path.exists(test_dir) else (None, None)

# Görselleri normalize edin
X_train = X_train / 255.0
if X_test is not None:
    X_test = X_test / 255.0
