import os
import numpy as np

# Eğitim veri seti yolunu tanımlayın
TRAIN_DATASET_PATH = r'C:\Users\yagiz\OneDrive\Masaüstü\kodlar\UnetsegmentationDeneme\unet-segmentation-project\data\train\BraTS2020_TrainingData\Training60'

# Verilen dizindeki .npy dosyalarını yükleyen fonksiyon
def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':  # Sadece .npy dosyalarını yükle
            image = np.load(os.path.join(img_dir, image_name)).astype(np.float32)
            images.append(image)
    images = np.array(images)
    return images

# Görüntü ve maske verilerini yükleyen ve batch halinde döndüren jeneratör fonksiyonu
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, dtype=np.float32):
    L = len(img_list)

    # Keras jeneratörünün sonsuz olması gerektiği için while true kullanıyoruz
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])  # Görüntüleri yükle
            Y = load_img(mask_dir, mask_list[batch_start:limit])  # Maskeleri yükle
            yield (X, Y)  # İki numpy array içeren bir tuple döndür

            batch_start += batch_size
            batch_end += batch_size

if __name__ == "__main__":
    # Jeneratörü test et
    from matplotlib import pyplot as plt
    import random

    # Eğitim görüntüleri ve maskeleri için dizinleri tanımla
    train_img_dir = os.path.join(TRAIN_DATASET_PATH, "input_data_60/train/images/")
    train_mask_dir = os.path.join(TRAIN_DATASET_PATH, "input_data_60/train/masks/")
    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    batch_size = 2

    # Eğitim görüntüleri için jeneratör oluştur
    train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                    train_mask_dir, train_mask_list, batch_size)

    # Jeneratörü doğrula.... Python 3'te next() __next__() olarak yeniden adlandırıldı
    img, msk = train_img_datagen.__next__()

    # Rastgele bir görüntü seç
    img_num = random.randint(0, img.shape[0] - 1)
    test_img = img[img_num]
    test_mask = msk[img_num]
    test_mask = np.argmax(test_mask, axis=3)

    # Rastgele bir dilim seç
    n_slice = random.randint(0, test_mask.shape[2])
    plt.figure(figsize=(12, 8))

    # Görüntü ve maskeyi görselleştir
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