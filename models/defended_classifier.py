import torch.nn as nn
from torchvision.models import resnet

from models.classifier import Classifier


class DefendedClassifier(Classifier):

    def _build_model(self, architecture, pretrained):

        architecture = architecture.lower().replace('-', '').replace('_', '').strip()

        if architecture == 'resnet50':

            weights = resnet.ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet.resnet50(weights=weights)

            # DEFENDED HEAD WITH DROPOUT
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, self.num_classes)
            )

            return model

        else:
            raise RuntimeError(
                f'Defended version only implemented for {architecture}'
            )