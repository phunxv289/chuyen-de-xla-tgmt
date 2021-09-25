from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from transformers import ViTModel


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
        model_logits = model_logits.squeeze()
        model_logits = self.dropout(model_logits)

        age_logits = self.age_output(model_logits)
        gender_logits = self.gender_output(model_logits)

        return age_logits, gender_logits


class EfficientNetCustom(nn.Module):
    def __init__(self, drop_out, num_age_class, num_gender_classs):
        super(EfficientNetCustom, self).__init__()
        base_model = EfficientNet.from_pretrained('efficientnet-b5')
        self.base_model, feat_size = remove_last_layer(base_model)
        self.dropout = nn.Dropout(drop_out)

        self.age_output = nn.Linear(feat_size, num_age_class)
        self.gender_output = nn.Linear(feat_size, num_gender_classs)

    def forward(self, batch_img):
        model_logits = self.base_model(batch_img)
        model_logits = model_logits.squeeze()
        model_logits = self.dropout(model_logits)

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
        model_logits = model_logits.squeeze()
        model_logits = self.dropout(model_logits)

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
