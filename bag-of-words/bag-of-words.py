from typing import List, Dict
import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vocab_dict: Dict = {v: id for id, v in enumerate(vocab)}
    bow = np.zeros((len(vocab),), dtype=np.int64)
    for token in tokens:
        token_id = vocab_dict.get(token, -1)
        if token_id == -1:
            continue
        else:
            bow[token_id] += 1

    return bow
    