train_file = 'data/train_data.tsv'
test_file = 'data/test_data.tsv'

label_column = 'answer'

unk_token = "<UNK>"  # "[UNK]"
sep_token = "<SEP>"  # "[SEP]"
pad_token = "<PAD>"  # "[PAD]"

unk_token_id = 1
sep_token_id = 3
pad_token_id = 0

# dictionary_file = 'data/tokenizer-vocab.txt'
tokenizer_path = 'models/tokenizer.model'
model_file = 'models/best-model.pth'
