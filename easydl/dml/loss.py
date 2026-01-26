import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss is a loss function for training a model to embed images into a high-dimensional space where the distance between the embeddings of different classes is maximized.
    It is a variant of the Proxy NCA loss, which is a loss function for training a model to embed images into a high-dimensional space where the distance between the embeddings of different classes is minimized.
    The original paper is https://arxiv.org/abs/2003.13911
    """

    def __init__(self, num_classes, embedding_dim, margin=0.1, alpha=32):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings, labels):
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        # Compute cosine similarity between embeddings and proxies
        sim_matrix = embeddings @ proxies.T  # (B, C)

        labels_one_hot = (
            F.one_hot(labels, num_classes=self.num_classes)
            .float()
            .to(sim_matrix.device)
        )

        pos_mask = labels_one_hot > 0
        neg_mask = ~pos_mask

        pos_term = torch.logsumexp(
            -self.alpha * (sim_matrix - self.margin) * pos_mask.float(), dim=1
        )
        neg_term = torch.logsumexp(
            self.alpha * (sim_matrix + self.margin) * neg_mask.float(), dim=1
        )

        loss = (pos_term + neg_term).mean()
        return loss


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """
    Implementation of ArcFace (Additive Angular Margin Loss).
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        """
        Args:
            in_features: Size of the input embedding (e.g., 512)
            out_features: Number of classes (identities)
            s: Norm of input feature (scale factor). Default is 64.0.
            m: Margin. Default is 0.50.
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # The weights are the "class centers" (one vector per class)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants for efficiency and stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Threshold for numerical stability (to handle angles > pi)
        # If cos(theta) <= th, we switch to a safe Taylor expansion approximation
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # 1. Normalize weights and input embeddings (L2 Norm)
        #    Result: cosine similarity matrix (batch_size, out_features)
        #    Range: [-1, 1]
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))

        # 2. Calculate sine of the angle (needed for trig identity: cos(a+b) = cos(a)cos(b) - sin(a)sin(b))
        #    Clamp ensures numerical stability (avoid negative values due to float precision)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # 3. Calculate cos(theta + m) using trig identity
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Handle numerical stability issues (when theta + m > pi)
        #    If cosine > th, use the computed phi. Otherwise, use a penalty term (cosine - mm).
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Convert labels to one-hot to apply margin ONLY to the target class
        #    We want to modify the logit for the Ground Truth class, leaving others untouched.
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 6. Combine: use phi (margin added) for target class, regular cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 7. Rescale by 's' to bring gradients into a range optimal for Softmax
        output *= self.s

        return output


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss is a loss function for training a model to embed images into a high-dimensional space where the distance between the embeddings of different classes is maximized.
    It is a variant of the ArcFace loss, which is a loss function for training a model to embed images into a high-dimensional space where the distance between the embeddings of different classes is maximized.
    The original paper is https://arxiv.org/abs/1801.07698
    """

    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.50):
        super().__init__()
        self.arcface = ArcFace(embedding_dim, num_classes, s, m)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.arcface(embeddings, labels)
        loss = self.loss_fn(logits, labels)
        return loss
