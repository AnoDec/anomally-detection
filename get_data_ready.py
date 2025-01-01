import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import random
import splitfolders
import os

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Eğitim veri seti yolunu tanımla
TRAIN_DATASET_PATH ='brats training path'

# FLAIR görüntüsünü yükle ve maksimum değerini yazdır
test_image_flair=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_047/BraTS20_Training_047_flair.nii').get_fdata()
print(test_image_flair.max())

# Görüntüyü ölçekle ve orijinal şekline geri döndür
test_image_flair=scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

# Diğer görüntüleri yükle ve ölçekle
test_image_t1=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_047/BraTS20_Training_047_t1.nii').get_fdata()
test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_047/BraTS20_Training_047_t1ce.nii').get_fdata()
test_image_t1ce=scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_047/BraTS20_Training_047_t2.nii').get_fdata()
test_image_t2=scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

# Maske görüntüsünü yükle ve veri tipini uint8 olarak değiştir
test_mask=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_047/BraTS20_Training_047_seg.nii').get_fdata()
test_mask=test_mask.astype(np.uint8)

# Maske değerlerini yeniden kodla (4'ü 3'e dönüştür)
print(np.unique(test_mask))  #0, 1, 2, 4 (0, 1, 2, 3 olarak yeniden kodlanacak)
test_mask[test_mask==4] = 3  #Maske değerlerini 4'ten 3'e yeniden ata
print(np.unique(test_mask)) 

# Rastgele bir dilim seç ve görüntüleri ve maskeyi çiz
n_slice=random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:,:,n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:,:,n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:,:,n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:,:,n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# Görüntüleri birleştir (FLAIR, T1ce, T2)
combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)

# Görüntüleri 64'e bölünebilir bir boyuta kırp
combined_x=combined_x[56:184, 56:184, 13:141] #128x128x128x4 boyutuna kırp

# Maskeyi de aynı şekilde kırp
test_mask = test_mask[56:184, 56:184, 13:141]

# Rastgele bir dilim seç ve kırpılmış görüntüleri ve maskeyi çiz
n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# Birleştirilmiş veriyi kaydet
COMBİNED_DATA_PATH = TRAIN_DATASET_PATH + '/combined.npy'
np.save(COMBİNED_DATA_PATH, combined_x)

# Kaydedilen görüntüyü doğrula
my_img=np.load(COMBİNED_DATA_PATH)

# Maskeyi kategorik hale getir
test_mask = to_categorical(test_mask, num_classes=4)

# Gerekli klasörleri oluştur
os.makedirs(os.path.join(TRAIN_DATASET_PATH, 'input_data_3channels/images'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DATASET_PATH, 'input_data_3channels/masks'), exist_ok=True)

# Görüntü ve maske dosyalarının listesini al
t2_list = sorted(glob.glob(TRAIN_DATASET_PATH+'/*/*t2.nii'))
t1ce_list = sorted(glob.glob(TRAIN_DATASET_PATH+'/*/*t1ce.nii'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH+'/*/*flair.nii'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+'/*/*seg.nii'))

# Her bir görüntü ve maske için işlemleri gerçekleştir
for img in range(len(t2_list)):   # Tüm listeler aynı boyutta olduğu için t1_list kullanılıyor
    print("Now preparing image and masks number: ", img)
      
    # Görüntüleri yükle ve ölçekle
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    # Maskeyi yükle ve yeniden kodla
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  # Maske değerlerini 4'ten 3'e yeniden ata
    
    # Görüntüleri birleştir
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
     
    # Görüntüleri 64'e bölünebilir bir boyuta kırp
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    # Maske değerlerinin dağılımını kontrol et
    val, counts = np.unique(temp_mask, return_counts=True)

    # En az %1 faydalı hacim varsa görüntüleri ve maskeyi kaydet
    if (1 - (counts[0]/counts.sum())) > 0.01:  # En az %1 faydalı hacim varsa
        print("Save Me")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save(os.path.join(TRAIN_DATASET_PATH, 'input_data_3channels/images', f'image_{img}.npy'), temp_combined_images)
        np.save(os.path.join(TRAIN_DATASET_PATH, 'input_data_3channels/masks', f'mask_{img}.npy'), temp_mask)
        
    else:
        print("I am useless")   
   
# Girdi klasörünü ve çıktı klasörünü tanımla
input_folder = TRAIN_DATASET_PATH + '/input_data_3channels/'
output_folder = TRAIN_DATASET_PATH + '/input_data_60/'

# Veriyi belirli bir oranda böl
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.70, .30), group_prefix=None) # Varsayılan değerler
