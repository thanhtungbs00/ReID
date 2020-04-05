import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchvision.models as models

class Resnet18Embedding(nn.Module):
    def __init__(self, embedding_dimension=128, pretrained=False, normalized = True):
        super(Resnet18Embedding, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.normalized = normalized
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        self.fine_tuning = True

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        embedding = self.model(images)
        if self.normalized:
          embedding = 10 * self.l2_norm(embedding)
        return embedding