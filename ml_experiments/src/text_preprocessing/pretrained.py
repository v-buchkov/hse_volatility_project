import numpy as np


def get_embedding_for_pretrained(lemmas, model, embedding_size=300):
    res = np.zeros(embedding_size)
    cnt = 0
    for word in lemmas.split():
        if word in model:
            res += np.array(model[word])
            cnt += 1
    if cnt:
        res = res / cnt
    return res
