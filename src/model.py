import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, hidden_dim, target_size, padding_idx,
                 fc1=1024, fc2=128, dropout_rate=0.1, topk=3):
        super(Classifier, self).__init__()
        self.topk = topk
        self.emb = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, bidirectional=True)  # , num_layers=1, dropout=dropout_rate)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim * 4 * self.topk, fc1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2, target_size)
        )

    def forward(self, tokens):
        h = self.emb(tokens)
        h, lstm_state = self.lstm(h)
        maxp = torch.topk(h, self.topk, 1, largest=True)[0].view((h.shape[0], -1))
        minp = torch.topk(h, self.topk, 1, largest=False)[0].view((h.shape[0], -1))
        h = torch.cat([maxp, minp], 1)
        return self.clf(h)
