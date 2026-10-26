import torch.nn.functional as F


def cosine_similarity(embedding_a, embedding_b):
    similarity = F.cosine_similarity(
        embedding_a,
        embedding_b,
        dim=1
    )

    return similarity