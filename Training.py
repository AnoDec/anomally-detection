TRAIN_DATASET_PATH = 'Brats için path'

import os
import numpy as np
from data_generator import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random
from tensorflow.keras import backend as K
import segmentation_models_3D as sm
from keras.metrics import MeanIoU
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.models import load_model

# GPU belleği büyümesini ayarla
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Varsayılan float türünü float32 olarak ayarla
tf.keras.backend.set_floatx('float32')

# Eğitim görüntüleri ve maskeleri için dizinleri tanımla
train_img_dir = TRAIN_DATASET_PATH + "/input_data_60/train/images/"
train_mask_dir = TRAIN_DATASET_PATH + "/input_data_60/train/masks/"

# Doğrulama görüntüleri ve maskeleri için dizinleri tanımla
val_img_dir = TRAIN_DATASET_PATH + "/input_data_60/val/images/"
val_mask_dir = TRAIN_DATASET_PATH + "/input_data_60/val/masks/"

# Tüm eğitim görüntülerini ve maskelerini listele
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

# Tüm doğrulama görüntülerini ve maskelerini listele
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Eğitim ve doğrulama için batch boyutunu ayarla
batch_size = 5

# Eğitim ve doğrulama için görüntü veri üreticilerini başlat
train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                              val_mask_dir, val_mask_list, batch_size)

# Bir batch görüntü ve maske alarak üreticiyi doğrula
img, msk = train_img_datagen.__next__()

# Görselleştirme için batch'ten rastgele bir görüntü ve maske seç
img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

# Görselleştirme için 3D görüntü ve maskeden rastgele bir dilim seç
n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

# Görüntünün farklı kanallarını ve karşılık gelen maskeyi çiz
plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

# Eğitim için kayıp fonksiyonu, metrikler ve optimizasyon tanımla
wt0, wt1, wt2, wt3 = np.float32(0.25), np.float32(0.25), np.float32(0.25), np.float32(0.25)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3], dtype=np.float32))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Özel Dice katsayısı metriğini tanımla
def dice_metric(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

# Özel IoU metriğini tanımla
def iou_metric(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

# Tüm metrikleri hesaplayan fonksiyon
def calculate_all_metrics(y_true, y_pred):
    """Tüm segmentasyon metriklerini hesapla"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    sensitivity = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    # Özgüllüğü hesapla
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    specificity = tn / (tn + fp + 1e-7)
    
    # IoU'yu hesapla
    intersection = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    union = np.sum((y_true_flat == 1) | (y_pred_flat == 1))
    iou = intersection / (union + 1e-7)
    
    # Dice katsayısını hesapla
    dice = 2 * intersection / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-7)
    
    return {
        'f1': f1,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'iou': iou,
        'dice': dice
    }
# Kullanılacak metrikler
metrics = [
    'accuracy',
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5),
    'precision',
    'recall',
    iou_metric,
    dice_metric
]
# Eğitim ve doğrulama metriklerini çizen fonksiyon
def plot_metrics(history, metric_name):
    """Eğitim ve doğrulama metriklerini çiz"""
    plt.figure(figsize=(10, 6))
    plt.plot(history[{metric_name}], 'y', label=f'Train {metric_name}')
    plt.plot(history[f'val_{metric_name}'], 'r', label=f'Validation{metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# Kullanılacak metrikler


# Öğrenme oranı ve optimizer tanımla
LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Eğitim adımlarını ve doğrulama adımlarını hesapla
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

from Unet_model import simple_unet_model

# U-Net modelini başlat
model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

# Modeli tanımlanan optimizer, kayıp ve metriklerle derle
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

# Modeli eğit
epochs = 5
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

# Eğitilmiş modeli kaydet
model_filename = f'saved_models/brats_3d_{epochs}epochs_simple_unet_weighted_dice.hdf5'
model.save(model_filename)

# Tüm metrikleri çiz
metric_names = ['f1', 'iou', 'dice', 'precision', 'sensitivity', 'specificity']
for metric in metric_names:
    plot_metrics(history.history, [metric])

# Tahmin veya eğitime devam etmek için modeli yükle
my_model = load_model(model_filename, 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5), 'f_score': sm.metrics.FScore(threshold=0.5)})
my_model.compile(optimizer=keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)

# Eğitime devam et
history2 = my_model.fit(
    train_img_datagen,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    verbose=1,
    validation_data=val_img_datagen,
    validation_steps=val_steps_per_epoch
)


# Birleştirilmiş metrikleri kaydet
combined_metrics = {
    metric: history.history[metric] + history2.history[metric]
    for metric in metric_names
}

# Test veri setinden bir batch görüntü üzerinde IoU'yu doğrula


test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                               val_mask_dir, val_mask_list, batch_size)

# Bir batch test görüntü ve maske al
test_image_batch, test_mask_batch = test_img_datagen.__next__()

# Test batch üzerinde tahmin yap
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

# Ortalama IoU'yu hesapla
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Tek bir test görüntüsü üzerinde tahmin yap
img_num = 42
test_img = np.load(TRAIN_DATASET_PATH + "/input_data_60/val/images/image_" + str(img_num) + ".npy")
test_mask = np.load(TRAIN_DATASET_PATH + "/input_data_60/val/masks/mask_" + str(img_num) + ".npy")
test_mask_argmax = np.argmax(test_mask, axis=3)

# Tahmin için test görüntüsünü hazırla
test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.show()
