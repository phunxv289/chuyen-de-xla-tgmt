import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from os.path import join
from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from transformers import ViTModel
import os
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F

RESOURCE_PATH = './resources'
CHECKPOINT_FOLDER = join(RESOURCE_PATH, 'checkpoint')
CHECKPOINT_PATTERN = join(RESOURCE_PATH, '{}_{}.pt')  # model_signature, acc_score
AGE_BINS = [12, 19, 26, 33, 39]
AGE_LABELS = [0, 1, 2, 3]
GENDER_MAPPER = {'male': 1, 'female': 0}
PATIENCE = 5
THRESHOLD = 0.5


class AgeGenderDataset(Dataset):
    def __init__(self, data_file, preprocess):
        self.preprocess = preprocess
        self.label_df = self.load_data_file(data_file)
        self.img_paths = self.label_df['Path'].tolist()
        self.age_labels = self.label_df['age_label'].tolist()
        self.gender_labels = self.label_df['gender_label'].tolist()
        self.data_size = len(self.label_df)

    def load_data_file(self, data_file):
        df = pd.read_csv(data_file)
        df = df.dropna()
        df = df[df['Type'] != 'log']
        print('Dataset size: %s ' % len(df))
        df['age_label'] = df['Age'].astype(int)
        df['gender_label'] = df['Gender'].astype(int)
        return df

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path)
        img = self.preprocess(img)
        age_label = self.age_labels[idx]
        gender_label = self.gender_labels[idx]

        return img, age_label, gender_label


def remove_last_layer(model):
    """ remove last layer of the base model and return the last feature """
    last_layer = list(model.children())[-1]
    feat_size = last_layer.in_features
    module_list = nn.Sequential(*list(model.children())[:-1])
    return module_list, feat_size


class ResNetCustom(nn.Module):
    def __init__(self, drop_out, num_age_class, num_gender_classs):
        super(ResNetCustom, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.base_model, feat_size = remove_last_layer(base_model)
        self.dropout = nn.Dropout(drop_out)

        self.age_output = nn.Linear(feat_size, num_age_class)
        self.gender_output = nn.Linear(feat_size, num_gender_classs)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        model_logits = self.dropout(model_logits)

        age_logits = self.age_output(model_logits)
        gender_logits = self.gender_output(model_logits)

        return age_logits, gender_logits


def customize_efficient_net(model):
    feat_size = model._fc.in_features
    model._fc = nn.Identity()
    model._swish = nn.Identity()
    return model, feat_size


class EfficientNetCustom(nn.Module):
    def __init__(self, drop_out, num_age_class, num_gender_classs):
        super(EfficientNetCustom, self).__init__()
        base_model = EfficientNet.from_pretrained('efficientnet-b1')
        self.base_model, feat_size = customize_efficient_net(base_model)
        self.dropout = nn.Dropout(drop_out)

        self.age_output = nn.Linear(feat_size, num_age_class)
        self.gender_output = nn.Linear(feat_size, num_gender_classs)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        # model_logits = self.dropout(model_logits)

        age_logits = self.age_output(model_logits)
        gender_logits = self.gender_output(model_logits)

        return age_logits, gender_logits


class VisionTrasnformerCustom(nn.Module):
    def __init__(self, drop_out, num_age_class, num_gender_classs):
        super(VisionTrasnformerCustom, self).__init__()
        self.base_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        feat_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(drop_out)

        self.age_output = nn.Linear(feat_size, num_age_class)
        self.gender_output = nn.Linear(feat_size, num_gender_classs)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        model_logits = self.dropout(model_logits.pooler_output)

        age_logits = self.age_output(model_logits)
        gender_logits = self.gender_output(model_logits)

        return age_logits, gender_logits


class AgeGenderModel(nn.Module):
    def __init__(self, model_signature, drop_out, num_age_class, num_gender_classs):
        """
            Build model by provided model signature.
            :param model_signature: string to identify which model to use. ['efficient-net', 'res-net', 'dense-net']
        """
        super(AgeGenderModel, self).__init__()
        self.model_signature = model_signature
        assert model_signature in ['efficient-net', 'res-net', 'vision-transformer']
        if model_signature == 'efficient-net':
            self.model = EfficientNetCustom(drop_out, num_age_class, num_gender_classs)
        elif model_signature == 'res-net':
            self.model = ResNetCustom(drop_out, num_age_class, num_gender_classs)
        else:
            self.model = VisionTrasnformerCustom(drop_out, num_age_class, num_gender_classs)

    def forward(self, img_batch):
        """
        :param img_batch: batch of images. [N x C x W x H]
        :return: image logits with the shape of [N]
        """
        age_logits, gender_logits = self.model(img_batch)
        return age_logits, gender_logits

    def get_total_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Model: %s - total parameters: %s - trainable parameter: %s ' % (
            self.model_signature, total_params, trainable_params))


class AgeGenderTrainer:
    def __init__(self, model_signature, train_file, test_file, drop_out, batch_size, learn_rate, n_epochs,
                 num_age_class, num_gender_classs, max_checkpoints=3):
        self.model_signature = model_signature
        self.n_max_cp = max_checkpoints
        self.n_epochs = n_epochs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        train_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        nontrain_preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

        self.trainer_iter = AgeGenderDataset(train_file, train_preprocess)
        self.test_iter = AgeGenderDataset(test_file, nontrain_preprocess)

        self.train_loader = DataLoader(self.trainer_iter,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(self.test_iter,
                                      batch_size=batch_size,
                                      shuffle=False)
        n_train_step = len(self.train_loader) * n_epochs
        self.model = AgeGenderModel(model_signature, drop_out, num_age_class, num_gender_classs)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learn_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, len(self.train_loader),
                                                                      n_train_step)

    def perform_training(self):
        scaler = GradScaler()
        max_score = 0
        patient_count = 0
        for ep in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            p_bar = tqdm(len(self.train_loader), ncols=500)
            for idx, batch_data in enumerate(self.train_loader):
                imgs, age_label, gender_label = [w.to(self.device) for w in batch_data]
                self.optimizer.zero_grad()
                with autocast():
                    age_logits, gender_logits = self.model(imgs)
                    age_loss = self.criterion(age_logits, age_label)
                    gender_loss = self.criterion(gender_logits, gender_label)
                    loss = age_loss + gender_loss

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                total_loss += loss.item()
                p_bar.set_description('Epoch: %s - iteration %s - loss %.3f - avg_loss: %.3f - learn_rate %.5f' % (
                    ep + 1, idx + 1, loss.item(), total_loss / (idx + 1), self.scheduler.get_last_lr()[0]))

            print('--- Done training for epoch %s ' % ep)
            print('--- Perform evaluation ...')
            age_accuracy, gender_accuracy = self.evaluate(self.test_loader)
            acc_score = (age_accuracy + gender_accuracy) / 2
            if acc_score > max_score:
                print('Model performance improve from %.3f to %.3f. Saving model. ' % (max_score, acc_score))
                self.save_model(acc_score)
                max_score = acc_score
            else:
                patient_count += 1
                if patient_count > PATIENCE:
                    print('Model performance is not improved for the last %s epochs. Exiting. ' % PATIENCE)
                    break

    def evaluate(self, data_loader):
        self.model.eval()
        age_labels = []
        gender_labels = []

        age_predicts = []
        gender_predicts = []
        with torch.no_grad():
            for idx, (imgs, age_label, gender_label) in enumerate(data_loader):
                imgs = imgs.to(self.device)
                age_labels.extend(age_label.numpy().tolist())
                gender_labels.extend(gender_label.numpy().tolist())
                age_logits, gender_logits = self.model(imgs)
                age_logits = F.softmax(age_logits, dim=-1)
                gender_logits = F.softmax(gender_logits, dim=-1)

                age_predict = age_logits.topk(1)[1]
                gender_predict = gender_logits.topk(1)[1]

                age_predicts.extend(age_predict.cpu().numpy().tolist())
                gender_predicts.extend(gender_predict.cpu().numpy().tolist())

        print('Age prediction:')
        age_accuracy = accuracy_score(age_labels, age_predicts)
        print(classification_report(age_labels, age_predicts))
        print('----------------------------')
        print('Gender prediction')
        gender_accuracy = accuracy_score(gender_labels, gender_predicts)
        print(classification_report(gender_labels, gender_predicts))

        return age_accuracy, gender_accuracy

    def save_model(self, acc_score):
        checkpoint = {'model_cp': self.model.state_dict(),
                      'optim_cp': self.optimizer.state_dict(),
                      'sched_cp': self.scheduler.state_dict()}
        cp_path = CHECKPOINT_PATTERN.format(self.model_signature, acc_score)
        torch.save(checkpoint, cp_path)
        cp_files = [join(CHECKPOINT_FOLDER, w) for w in os.listdir(CHECKPOINT_FOLDER)]
        keep_cp = sorted(cp_files, reverse=True)[:self.n_max_cp]
        remove_cp = set(cp_files).difference(keep_cp)
        if len(remove_cp) > 0:
            os.remove(list(remove_cp)[0])
        print('Model {} saved successfully.'.format(cp_path))

    def load_model(self, cp_path):
        checkpoint = torch.load(cp_path)
        self.model.load_state_dict(checkpoint['model_cp'])
        self.optimizer.load_state_dict(checkpoint['optim_cp'])
        self.scheduler.load_state_dict(checkpoint['sched_cp'])

        print('Model load sucessfully.')


if __name__ == '__main__':
    model_signature = 'vision-transformer'
    img_folder = ''
    train_lb_path = ''
    test_lb_path = ''
    drop_out = 0.25
    batch_size = 32
    learn_rate = 0.005
    n_epochs = 10
    num_age_class = 5
    num_gender_classs = 2
    trainer = AgeGenderTrainer(model_signature, img_folder, train_lb_path, test_lb_path,
                               drop_out, batch_size, learn_rate, n_epochs, num_age_class, num_gender_classs,
                               max_checkpoints=3)
    trainer.perform_training()
