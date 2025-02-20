import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import datetime
import random

# 1. Define Vocabulary and Tokenization

INPUT_CHARS = list("0123456789/")  # Characters in input format "%m/%d/%Y"
OUTPUT_CHARS = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,-") # Characters in output format "%B %d, %Y"

INPUT_VOCAB = {char: idx for idx, char in enumerate(INPUT_CHARS)}
INPUT_VOCAB['<pad>'] = len(INPUT_VOCAB) # Add padding token
INPUT_VOCAB['<sos>'] = len(INPUT_VOCAB) # Add start of sequence token
INPUT_VOCAB['<eos>'] = len(INPUT_VOCAB) # Add end of sequence token
INPUT_VOCAB_SIZE = len(INPUT_VOCAB)

OUTPUT_VOCAB = {char: idx for idx, char in enumerate(OUTPUT_CHARS)}
OUTPUT_VOCAB['<pad>'] = len(OUTPUT_VOCAB) # Add padding token
OUTPUT_VOCAB['<sos>'] = len(OUTPUT_VOCAB) # Add start of sequence token
OUTPUT_VOCAB['<eos>'] = len(OUTPUT_VOCAB) # Add end of sequence token
OUTPUT_VOCAB_SIZE = len(OUTPUT_VOCAB)

INPUT_PAD_IDX = INPUT_VOCAB['<pad>']
OUTPUT_PAD_IDX = OUTPUT_VOCAB['<pad>']
INPUT_SOS_IDX = INPUT_VOCAB['<sos>']
OUTPUT_SOS_IDX = OUTPUT_VOCAB['<sos>']
INPUT_EOS_IDX = INPUT_VOCAB['<eos>']
OUTPUT_EOS_IDX = OUTPUT_VOCAB['<eos>']


def tokenize_input(text):
    return [INPUT_VOCAB.get(char, INPUT_VOCAB['<pad>']) for char in text]

def tokenize_output(text):
    return [OUTPUT_VOCAB.get(char, OUTPUT_VOCAB['<pad>']) for char in text]

def detokenize_output(tokens):
    reverse_output_vocab = {idx: char for char, idx in OUTPUT_VOCAB.items()}
    return "".join([reverse_output_vocab.get(token, '') for token in tokens if token not in [OUTPUT_PAD_IDX, OUTPUT_SOS_IDX, OUTPUT_EOS_IDX]])


# 2. Create Date Dataset

class DateTranslationDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.input_dates, self.output_dates = self._generate_date_pairs(num_samples)

    def _generate_date_pairs(self, num_samples):
        input_dates = []
        output_dates = []
        start_date = datetime.date(1900, 1, 1)
        end_date = datetime.date(2100, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days

        for _ in range(num_samples):
            random_number_of_days = random.randrange(days_between_dates)
            random_date = start_date + datetime.timedelta(days=random_number_of_days)

            input_date = random_date.strftime("%m/%d/%Y")
            output_date = random_date.strftime("%B %d, %Y")

            input_dates.append(input_date)
            output_dates.append(output_date)
        return input_dates, output_dates

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_dates[idx], self.output_dates[idx]

def collate_fn(batch):
    input_texts, output_texts = zip(*batch)

    input_tokens = [torch.tensor([INPUT_SOS_IDX] + tokenize_input(text) + [INPUT_EOS_IDX] + [INPUT_PAD_IDX]) for text in input_texts]
    output_tokens = [torch.tensor([OUTPUT_SOS_IDX] + tokenize_output(text) + [OUTPUT_EOS_IDX]) for text in output_texts]

    input_tokens_padded = pad_sequence(input_tokens, padding_value=INPUT_PAD_IDX, batch_first=True)
    output_tokens_padded = pad_sequence(output_tokens, padding_value=OUTPUT_PAD_IDX, batch_first=True)

    return input_tokens_padded, output_tokens_padded


# 3. Define Transformer Model

class DateTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()
        self.input_embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embed_dim, output_vocab_size)

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim


    def forward(self, src, trg, src_mask, trg_mask, memory_mask, src_padding_mask, trg_padding_mask, memory_padding_mask):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_embedded = self.pos_encoder(self.input_embedding(src)) # [batch_size, src_len, embed_dim]
        trg_embedded = self.pos_encoder(self.output_embedding(trg)) # [batch_size, trg_len, embed_dim]

        # Transformer layers expect (S, N, E) or (N, S, E) where S is sequence length, N is batch size, E is embedding dim.
        # batch_first=True in nn.Transformer makes it (N, S, E)

        memory = self.transformer.encoder(src_embedded, mask=src_mask, src_key_padding_mask=src_padding_mask) # [batch_size, src_len, embed_dim]
        output = self.transformer.decoder(trg_embedded, memory, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=memory_padding_mask) # [batch_size, trg_len, embed_dim]

        output = self.fc_out(output) # [batch_size, trg_len, output_vocab_size]
        return output

    def encode(self, src, src_mask, src_padding_mask):
        src_embedded = self.pos_encoder(self.input_embedding(src))
        memory = self.transformer.encoder(src_embedded, mask=src_mask, src_key_padding_mask=src_padding_mask)
        return memory

    def decode(self, trg, memory, trg_mask, trg_padding_mask, memory_mask, memory_padding_mask):
        trg_embedded = self.pos_encoder(self.output_embedding(trg))
        output = self.transformer.decoder(trg_embedded, memory, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=memory_padding_mask)
        return self.fc_out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Store pe as a buffer, not a trainable parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


def create_mask(src, trg, src_pad_idx, trg_pad_idx):
    src_seq_len = src.shape[1]
    trg_seq_len = trg.shape[1]

    trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_len).to(src.device) # Decoder mask, prevents peeking
    src_padding_mask = (src == src_pad_idx) # Mask padding tokens in source
    trg_padding_mask = (trg == trg_pad_idx) # Mask padding tokens in target
    return trg_mask, src_padding_mask, trg_padding_mask


# 4. Training Loop

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.1
MAX_SEQ_LEN = 30 # Maximum expected output sequence length

# Dataset and DataLoader
dataset = DateTranslationDataset(num_samples=50000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataset = DateTranslationDataset(num_samples=1000)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# Model, Optimizer, Loss
model = DateTransformer(INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=OUTPUT_PAD_IDX) # Ignore padding tokens in loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def remove_sos_add_pad(target_tensors, pad_token=OUTPUT_PAD_IDX):
    """
    Remove the <sos> token from batch padded target tensors and add a pad token at the end.

    Args:
        target_tensors (torch.Tensor): Batch of padded target tensors.
            Shape: (batch_size, sequence_length)
        pad_token (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: Batch of target tensors with <sos> token removed and pad token added.
            Shape: (batch_size, sequence_length)
    """
    # Remove the first token (assumed to be <sos>) from each sequence
    without_sos = target_tensors[:, 1:]
    
    # Create a tensor of pad tokens with shape (batch_size, 1)
    pad_column = torch.full((target_tensors.size(0), 1), pad_token, dtype=target_tensors.dtype, device=target_tensors.device)
    
    # Concatenate the pad column to the end of the sequences
    result = torch.cat([without_sos, pad_column], dim=1)
    
    return result


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)

        trg_input = trg[:, :-1] # Remove column to match the shape of the output
        trg_output = trg[:, 1:] # Remove <sos> token for target output

        trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, INPUT_PAD_IDX, OUTPUT_PAD_IDX)

        optimizer.zero_grad()
        output = model(src, trg_input, None, trg_mask, None, src_padding_mask, trg_padding_mask, src_padding_mask) # Memory masks are None as we are not explicitly using them in this simplified example.
        # output shape: [batch_size, trg_len-1, output_vocab_size]
        # trg_output shape: [batch_size, trg_len-1]

        output_reshape = output.contiguous().view(-1, model.output_vocab_size) # [batch_size * trg_len-1, output_vocab_size]
        trg_output_reshape = trg_output.contiguous().view(-1) # [batch_size * trg_len-1]

        loss = criterion(output_reshape, trg_output_reshape)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            trg_input =  trg[:, :-1] 
            trg_output = trg[:, 1:]

            trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, INPUT_PAD_IDX, OUTPUT_PAD_IDX)

            output = model(src, trg_input, None, trg_mask, None, src_padding_mask, trg_padding_mask, src_padding_mask)

            output_reshape = output.contiguous().view(-1, model.output_vocab_size)
            trg_output_reshape = trg_output.contiguous().view(-1)

            loss = criterion(output_reshape, trg_output_reshape)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Training loop
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
    val_loss = evaluate_epoch(model, val_dataloader, criterion, device)
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# 5. Translation / Inference Function

def translate_date(model, input_date_str, device, max_len=MAX_SEQ_LEN):
    model.eval()
    input_tokens = torch.tensor([INPUT_SOS_IDX] + tokenize_input(input_date_str) + [INPUT_EOS_IDX]).unsqueeze(0).to(device) # [1, src_len]
    src_padding_mask = (input_tokens == INPUT_PAD_IDX).to(device)

    memory = model.encode(input_tokens, None, src_padding_mask) # Encode the source

    trg_tokens = [OUTPUT_SOS_IDX] # Start with <sos> token
    for _ in range(max_len):
        trg_input = torch.tensor(trg_tokens).unsqueeze(0).to(device) # [1, current_trg_len]
        trg_mask = (nn.Transformer.generate_square_subsequent_mask(trg_input.size(1)).to(device))
        trg_padding_mask = (trg_input == OUTPUT_PAD_IDX).to(device)

        output = model.decode(trg_input, memory, trg_mask, trg_padding_mask, None, src_padding_mask) # [1, current_trg_len, output_vocab_size]
        # Get the last predicted token distribution
        next_token_probs = output[:, -1, :] # [1, output_vocab_size]
        next_token = next_token_probs.argmax(dim=-1).item() # Get token with highest probability

        if next_token == OUTPUT_EOS_IDX: # Stop if <eos> is predicted
            break
        trg_tokens.append(next_token)

    return detokenize_output(trg_tokens)


# 6. Test Translation

# Example usage
model.eval() # Set to evaluation mode for inference

input_date_examples = ["03/15/2023", "12/01/1999", "01/01/2024", "09/30/2000", "10/16/1985"]

for input_date in input_date_examples:
    predicted_output_date = translate_date(model, input_date, device)
    print(f"Input Date: {input_date}, Predicted Output Date: {predicted_output_date}")