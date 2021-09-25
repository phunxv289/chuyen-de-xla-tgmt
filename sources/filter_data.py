import os
import numpy as np
from PIL import Image
import face_detection
from tqdm.notebook import tqdm

from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.imgs_list = os.listdir(img_folder)
        self.data_size = len(self.imgs_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_file = self.imgs_list[idx]
        img_id = os.path.splitext(img_file)[0]
        img_file = os.path.join(self.img_folder, img_file)
        img = Image.open(img_file)

        return img, img_id


img_folder = ''
face_dataset = FaceDataset(img_folder)
face_loader = DataLoader(face_dataset, )

print(face_detection.available_detectors)
detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

label_mapper = {}
for imgs, img_ids in tqdm(face_loader, total=len(face_loader)):
    imgs = imgs.numpy().astype(np.uint8)
    detections = detector.batched_detect(imgs)
    face_presents = [True if w.size > 0 else False for w in detections]
    label_mapper.update(dict(zip(*(img_ids, face_presents))))
