from itertools import cycle

from torch.utils import data


class BinaryDataset(data.Dataset):
    def __init__(self, x, y):
        """
        :param x: pd.Series or pd.DataFrame
        :param y: pd.Series or pd.DataFrame
        """
        self.x = x.values
        self.y = y.astype(int).values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BinaryEvenDataset(BinaryDataset):
    def __init__(self, x, y):
        """
        Warning! When you put it to dataloader, use shuffle=True argument
        :param x: pd.Series or pd.DataFrame
        :param y: pd.Series or pd.DataFrame of bool
        """

        pos = x[y]
        neg = x[~y]
        if len(pos) > len(neg):
            big, small = pos, neg
            big_label, small_label = 1, 0
        else:
            small, big = pos, neg
            small_label, big_label = 1, 0

        self.small_label = small_label
        self.big_label = big_label
        self.big = cycle(big.values)
        self.small = small.values
        self.half_len = len(self.small)

    def __len__(self):
        return 2 * len(self.small)

    def __getitem__(self, idx):
        if idx >= self.half_len:
            return next(self.big), self.big_label
        else:
            return self.small[idx], self.small_label
