import mxnet as mx
import os
import cv2
from collections import defaultdict


idx_path = '/dataset/faces_webface_112x112/train.idx'
rec_path = '/dataset/faces_webface_112x112/train.rec'


output_dir = '/dataset/output_dataset'
authorized_dir = os.path.join(output_dir, 'authorized')
unauthorized_dir = os.path.join(output_dir, 'unauthorized')

os.makedirs(authorized_dir, exist_ok=True)
os.makedirs(unauthorized_dir, exist_ok=True)

# Czytnik MXNET
record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
keys = list(record.keys)

# Słownik służący do pogrupowania zdjęć
images_by_user = defaultdict(list)

for k in keys:
    header, img = mx.recordio.unpack(record.read_idx(k))
    header_int=int(header.label)
    if header_int > 200:
        break
    user_id = header_int
    images_by_user[user_id].append(img)

# Wybór użytkowników
all_users = sorted(images_by_user.keys())
authorized_users = all_users[:100]
unauthorized_users = all_users[100:200]

def save_images(user_list, user_type, base_dir, max_images=20):
    for i, user_id in enumerate(user_list):
        user_folder = os.path.join(base_dir, f"user_{i+1:05d}")
        os.makedirs(user_folder, exist_ok=True)
        images = images_by_user[user_id][:max_images]
        for j, img in enumerate(images):
            img_arr = mx.image.imdecode(img).asnumpy()
            filename = os.path.join(user_folder, f"img_{j+1:03d}.jpg")
            cv2.imwrite(filename, img_arr)

# Zapisz dane
save_images(authorized_users, 'authorized', authorized_dir)
save_images(unauthorized_users, 'unauthorized', unauthorized_dir)
