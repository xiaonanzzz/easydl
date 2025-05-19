import torch.nn as nn
import torch.nn.functional as F
import torch



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
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        # Compute cosine similarity between embeddings and proxies
        sim_matrix = embeddings @ proxies.T  # (B, C)

        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(sim_matrix.device)

        pos_mask = labels_one_hot > 0
        neg_mask = ~pos_mask

        pos_term = torch.logsumexp(-self.alpha * (sim_matrix - self.margin) * pos_mask.float(), dim=1)
        neg_term = torch.logsumexp(self.alpha * (sim_matrix + self.margin) * neg_mask.float(), dim=1)

        loss = (pos_term + neg_term).mean()
        return loss