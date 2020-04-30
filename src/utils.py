from functools import partial

import numpy as np
import torch
from torch import LongTensor
# Prepare sentences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def clue(df, sep_token, to_lower=True):
    """
    :param df: pd.DataFrame with columns 'ru_name' and 'eng_name'
    """
    tmp = df.apply(lambda r: '{} {} {}'.format(r["ru_name"], sep_token, r["eng_name"]), axis=1)
    if to_lower:
        tmp = tmp.apply(lambda r: r.lower())
    return tmp


def create_dataloader(data, batch_size, pad_id,
                      pin_memory=True, shuffle=True, num_of_workers=0):
    """Create DataLoader object for given data.
    """
    partial_collate = partial(my_collate, pad_id=pad_id)
    return DataLoader(data,
                      batch_size=batch_size,
                      collate_fn=partial_collate,
                      pin_memory=pin_memory,
                      drop_last=False,
                      shuffle=shuffle,
                      num_workers=num_of_workers)


def my_collate(batch, pad_id):
    """
    :param batch: batch to preprocess
    :return: preprocessed batch
    """
    tokens, targets = zip(*batch)
    tokens = [LongTensor(x) for x in tokens]
    # Pad sentences to max length.
    tokens = pad_sequence(tokens, padding_value=pad_id, batch_first=True)
    targets = LongTensor(targets).unsqueeze(1)
    return tokens, targets


def calc_accuracy(pred_tags, target):
    """Calculate the accuracy of the prediction.
    :param pred_tags: torch.Tensor 2D (one hot)
    :param target: tensor
    :return: accuracy in range [0;1]
    """
    _, pred = torch.max(pred_tags, 1)

    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    return correct.float().mean()
