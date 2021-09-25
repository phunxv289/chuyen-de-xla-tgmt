import os
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F
from sources.data_utils import AgeGenderDataset
from sources.model_utils import AgeGenderModel
from sources.config import *

PATIENCE = 5
THRESHOLD = 0.5


class AgeGenderTrainer:
    def __init__(self, model_signature, train_path, train_lb_path, test_path, test_lb_path,
                 drop_out, batch_size, learn_rate, n_epochs, num_age_class, num_gender_classs, max_checkpoints=3):
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

        self.trainer_iter = AgeGenderDataset(train_path, train_lb_path, train_preprocess)
        self.test_iter = AgeGenderDataset(test_path, test_lb_path, nontrain_preprocess)

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
        os.remove(list(remove_cp)[0])
        print('Model {} saved successfully.'.format(cp_path))

    def load_model(self, cp_path):
        checkpoint = torch.load(cp_path)
        self.model.load_state_dict(checkpoint['model_cp'])
        self.optimizer.load_state_dict(checkpoint['optim_cp'])
        self.scheduler.load_state_dict(checkpoint['sched_cp'])

        print('Model load sucessfully.')
