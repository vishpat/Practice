import random
from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchview import draw_graph
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
# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


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
        for char in out:
            output_vocab.add(char)

    input_vocab = sorted(list(input_vocab))
    output_vocab = sorted(list(output_vocab))

    # Add special tokens
    input_vocab = ["<PAD>", "<SOS>", "<EOS>"] + input_vocab
    output_vocab = ["<PAD>", "<SOS>", "<EOS>"] + output_vocab

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
        input_tensor = self.tensorize(input_date, self.input_vocab_to_idx)
        output_tensor = self.tensorize(output_date, self.output_vocab_to_idx)
        return input_tensor, output_tensor

    def tensorize(self, date_str, vocab_to_idx):
        tensor = [vocab_to_idx["<SOS>"]]
        for char in date_str:
            tensor.append(vocab_to_idx[char])
        tensor.append(vocab_to_idx["<EOS>"])
        tensor += [vocab_to_idx["<PAD>"]] * (self.max_len - len(tensor))
        return torch.tensor(tensor).to(device)


def tensor_to_string(tensor, idx_to_vocab):
    ids = tensor.cpu().numpy()
    return "".join([idx_to_vocab[idx] for idx in ids])


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


class TransFormer(nn.Module):
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
        super(TransFormer, self).__init__()
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

    def forward(self, src, tgt):
        src = self.pos_encoder(self.input_embedding(src))
        tgt = self.pos_encoder(self.output_embedding(tgt))
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

def train_transformer(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        output = output[:, :-1, :].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def validate_transformer(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            output = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[:, :-1, :].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


if __name__ == "__main__":
    date_pairs = generate_date_pairs(20000)
    input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab = (
        create_vocab(date_pairs)
    )

    print("Input Vocab Size:", len(input_vocab_to_idx))
    print("Output Vocab Size:", len(output_vocab_to_idx))

    train_set, test_set = train_test_split(date_pairs, test_size=0.2, train_size=0.8)
    test_set, val_set = train_test_split(test_set, test_size=0.5, train_size=0.5)
    train_dataset = DateDataSet(train_set, input_vocab_to_idx, output_vocab_to_idx)
    val_dataset = DateDataSet(val_set, input_vocab_to_idx, output_vocab_to_idx)
    test_dataset = DateDataSet(test_set, input_vocab_to_idx, output_vocab_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = TransFormer(
        len(input_vocab_to_idx),
        len(output_vocab_to_idx),
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
    )
    
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab_to_idx["<PAD>"])
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_transformer(model, train_loader, optimizer, criterion)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")
        val_loss = validate_transformer(model, val_loader, criterion)
        print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "date_translator.pth")

    if os.path.exists("date_translator.pth"):
        model.load_state_dict(torch.load("date_translator.pth"))
        test_loss = validate_transformer(model, test_loader, criterion)
        print(f"Test Loss: {test_loss}")
