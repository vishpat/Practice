import random
from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import math
import os

# Initialize Hyperparameters
embed_dim = 128
num_heads = 8
hidden_dim = 512
num_layers = 4
dropout = 0.1
learning_rate = 0.0001
num_epochs = 20
batch_size = 64
max_len = 20
# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"


def generate_date_pairs(num_samples):
    """Generates date pairs in both input and output format."""
    date_pairs = []
    for _ in range(num_samples):
        year = random.randint(1900, 2100)
        month = random.randint(1, 12)
        day = random.randint(
            1, 28
        )  # Simplified for simplicity, avoid end of month issues
        try:
            date_obj = date(year, month, day)
        except ValueError:  # This will catch invalid dates like Feb 29 in non leap year
            continue

        input_date = date_obj.strftime("%m/%d/%Y")
        output_date = date_obj.strftime("%B %d, %Y")
        date_pairs.append((input_date, output_date))
    return date_pairs


def create_vocab(date_pairs):
    """Creates vocabulary based on the data."""
    input_vocab = set()
    output_vocab = set()
    for inp, out in date_pairs:
        for char in inp:
            input_vocab.add(char)

        month, day, year = out.split(" ")
        output_vocab.add(month)
        for char in day:
            output_vocab.add(char)
        for char in year:
            output_vocab.add(char)

    input_vocab = sorted(list(input_vocab))
    output_vocab = sorted(list(output_vocab))
    # Add special tokens
    input_vocab = [PAD, SOS, EOS] + input_vocab
    output_vocab = [PAD, SOS, EOS] + [" "] + output_vocab
    input_vocab_to_idx = {token: idx for idx, token in enumerate(input_vocab)}
    input_idx_to_vocab = {idx: token for idx, token in enumerate(input_vocab)}
    output_vocab_to_idx = {token: idx for idx, token in enumerate(output_vocab)}
    output_idx_to_vocab = {idx: token for idx, token in enumerate(output_vocab)}
    return (
        input_vocab_to_idx,
        input_idx_to_vocab,
        output_vocab_to_idx,
        output_idx_to_vocab,
    )


INPUT_PAD_IDX = 0
INPUT_SOS_IDX = 1
INPUT_EOS_IDX = 2

OUTPUT_PAD_IDX = 0
OUTPUT_SOS_IDX = 1
OUTPUT_EOS_IDX = 2


def tensorize_input(date_str, vocab_to_idx):
    tensor = [vocab_to_idx[SOS]]
    for char in date_str:
        tensor.append(vocab_to_idx[char])
    tensor.append(vocab_to_idx[EOS])
    tensor += [vocab_to_idx[PAD]] * (max_len - len(tensor))
    return torch.tensor(tensor).to(device)


def tensorize_output(date_str, vocab_to_idx):
    month, day, year = date_str.split(" ")
    tensor = [vocab_to_idx[SOS]]
    tensor.append(vocab_to_idx[month])
    tensor.append(vocab_to_idx[" "])
    tensor += [vocab_to_idx[char] for char in day]
    tensor.append(vocab_to_idx[" "])
    tensor += [vocab_to_idx[char] for char in year]
    tensor.append(vocab_to_idx[EOS])
    tensor += [vocab_to_idx[PAD]] * (max_len - len(tensor) + 1)
    return torch.tensor(tensor).to(device)


class DateDataSet(Dataset):
    def __init__(self, date_pairs, input_vocab_to_idx, output_vocab_to_idx, max_len=20):
        self.max_len = max_len
        self.date_pairs = date_pairs
        self.input_vocab_to_idx = input_vocab_to_idx
        self.output_vocab_to_idx = output_vocab_to_idx

    def __len__(self):
        return len(self.date_pairs)

    def __getitem__(self, idx):
        input_date, output_date = self.date_pairs[idx]
        input_tensor = tensorize_input(input_date, self.input_vocab_to_idx)
        output_tensor = tensorize_output(output_date, self.output_vocab_to_idx)
        return input_tensor, output_tensor


def tensor_to_string(tokens, idx_to_vocab):
    return "".join([idx_to_vocab[idx] for idx in tokens])


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=20):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class DateTransFormer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super(DateTransFormer, self).__init__()
        self.input_embedding = Embedding(input_vocab_size, embed_dim)
        self.output_embedding = Embedding(output_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(embed_dim, output_vocab_size)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        memory_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_padding_mask,
    ):
        src = self.pos_encoder(self.input_embedding(src))
        tgt = self.pos_encoder(self.output_embedding(tgt))

        memory = self.transformer.encoder(
            src, src_mask, src_key_padding_mask=src_padding_mask
        )
        output = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        output = self.fc(output)
        return output

    def encode(self, src, src_mask, src_padding_mask):
        src_embedded = self.pos_encoder(self.input_embedding(src))
        memory = self.transformer.encoder(
            src_embedded, mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        return memory

    def decode(
        self, trg, memory, trg_mask, trg_padding_mask, memory_mask, memory_padding_mask
    ):
        trg_embedded = self.pos_encoder(self.output_embedding(trg))
        output = self.transformer.decoder(
            trg_embedded,
            memory,
            tgt_mask=trg_mask,
            tgt_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.fc(output)


def create_mask(src, trg):
    trg_seq_len = trg.shape[1]

    trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_len).to(
        src.device
    )  # Decoder mask, prevents peeking
    src_padding_mask = src == input_vocab_to_idx[PAD]  # Mask padding tokens in source
    trg_padding_mask = trg == output_vocab_to_idx[PAD]  # Mask padding tokens in target
    return trg_mask, src_padding_mask, trg_padding_mask


def train_transformer(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for _, (src, tgt) in enumerate(iterator):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask, src_padding_mask, trg_padding_mask = create_mask(src, tgt_input)
        optimizer.zero_grad()
        output = model(
            src,
            tgt_input,
            None,
            tgt_mask,
            None,
            src_padding_mask,
            trg_padding_mask,
            src_padding_mask,
        )
        output = output.contiguous().view(-1, model.output_vocab_size)
        tgt = tgt_output.contiguous().view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def validate_transformer(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (src, tgt) in enumerate(iterator):
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask, src_padding_mask, trg_padding_mask = create_mask(src, tgt_input)
            output = model(
                src,
                tgt_input,
                None,
                tgt_mask,
                None,
                src_padding_mask,
                trg_padding_mask,
                src_padding_mask,
            )
            output = output.contiguous().view(-1, model.output_vocab_size)
            tgt = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


date_file = "dates.txt"


def load_date_pairs():
    date_pairs = []
    if os.path.exists(date_file):
        print("Loading date pairs from file")
        with open(date_file, "r") as f:
            for line in f:
                a, b, c = line.strip().split(",")
                date_pairs.append(tuple((a, f"{b},{c}")))
    else:
        date_pairs = generate_date_pairs(20000)
        print("Saving date pairs to file")
        with open(date_file, "w") as f:
            for pair in date_pairs:
                f.write(f"{pair[0]},{pair[1]}\n")
    return date_pairs


def translate_date(model, input_date_str, device):
    model.eval()
    input_tokens = tensorize_input(input_date_str, input_vocab_to_idx).unsqueeze(
        0
    )  # [1, input_seq_len]
    src_padding_mask = (input_tokens == INPUT_PAD_IDX).to(device)

    memory = model.encode(input_tokens, None, src_padding_mask)  # Encode the source

    trg_tokens = [OUTPUT_SOS_IDX]  # Start with <sos> token
    for _ in range(max_len):
        trg_input = (
            torch.tensor(trg_tokens).unsqueeze(0).to(device)
        )  # [1, current_trg_len]
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(1)).to(
            device
        )
        trg_padding_mask = (trg_input == OUTPUT_PAD_IDX).to(device)

        output = model.decode(
            trg_input, memory, trg_mask, trg_padding_mask, None, src_padding_mask
        )  # [1, current_trg_len, output_vocab_size]
        # Get the last predicted token distribution
        next_token_probs = output[:, -1, :]  # [1, output_vocab_size]
        next_token = next_token_probs.argmax(
            dim=-1
        ).item()  # Get token with highest probability

        if next_token == OUTPUT_EOS_IDX:  # Stop if <eos> is predicted
            break
        trg_tokens.append(next_token)

    return tensor_to_string(trg_tokens[1:], output_idx_to_vocab)


if __name__ == "__main__":
    date_pairs = load_date_pairs()
    input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab = (
        create_vocab(date_pairs)
    )

    print(f"Input Vocab Size {len(input_vocab_to_idx)}")
    print(f"Output Vocab Size {len(output_vocab_to_idx)}")

    train_set, test_set = train_test_split(date_pairs, test_size=0.2, train_size=0.8)
    test_set, val_set = train_test_split(test_set, test_size=0.5, train_size=0.5)
    train_dataset = DateDataSet(train_set, input_vocab_to_idx, output_vocab_to_idx)
    val_dataset = DateDataSet(val_set, input_vocab_to_idx, output_vocab_to_idx)
    test_dataset = DateDataSet(test_set, input_vocab_to_idx, output_vocab_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = DateTransFormer(
        len(input_vocab_to_idx),
        len(output_vocab_to_idx),
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab_to_idx[PAD])

    # Initialize the optimizer
    if not os.path.exists("date_translator.pth"):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_transformer(model, train_loader, optimizer, criterion)
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}")
            val_loss = validate_transformer(model, val_loader, criterion)
            print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "date_translator.pth")

    model.load_state_dict(torch.load("date_translator.pth"))
    model.eval()  # Set to evaluation mode for inference

    input_date_examples = [
        "03/15/2023",
        "12/01/1999",
        "01/01/2024",
        "09/30/2000",
        "10/10/1985",
    ]

    for input_date in input_date_examples:
        predicted_output_date = translate_date(model, input_date, device)
        print(
            f"Input Date: {input_date}, Predicted Output Date: {predicted_output_date}"
        )
#    model.load_state_dict(torch.load("date_translator.pth"))
#    test_loss = validate_transformer(model, test_loader, criterion)
#    print(f"Test Loss: {test_loss}")
#    rand_idx = random.randint(0, len(train_dataset.date_pairs))
#    input_date, output_date = train_dataset.date_pairs[rand_idx]
#    print(f"{input_date} {output_date}")
#
