import random
from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os

# Mapping for numeric month to string month
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def generate_date_pairs(num_samples):
    """Generates date pairs in both input and output format."""
    date_pairs = []
    for _ in range(num_samples):
        year = random.randint(1900, 2100)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Simplified for simplicity, avoid end of month issues
        try:
            date_obj = date(year, month, day)
        except ValueError: # This will catch invalid dates like Feb 29 in non leap year
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
    input_vocab = ['<PAD>', '<SOS>', '<EOS>'] + input_vocab
    output_vocab = ['<PAD>', '<SOS>', '<EOS>'] + output_vocab

    input_vocab_to_idx = {token: idx for idx, token in enumerate(input_vocab)}
    input_idx_to_vocab = {idx: token for idx, token in enumerate(input_vocab)}
    output_vocab_to_idx = {token: idx for idx, token in enumerate(output_vocab)}
    output_idx_to_vocab = {idx: token for idx, token in enumerate(output_vocab)}

    return (input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab)


class DateDataset(Dataset):
    """Custom dataset for date pairs"""
    def __init__(self, date_pairs, input_vocab_to_idx, output_vocab_to_idx, max_len=20):
        self.date_pairs = date_pairs
        self.input_vocab_to_idx = input_vocab_to_idx
        self.output_vocab_to_idx = output_vocab_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.date_pairs)

    def __getitem__(self, idx):
        input_date, output_date = self.date_pairs[idx]

        # Encode the input and output sequences
        input_encoded = [self.input_vocab_to_idx['<SOS>']] + [self.input_vocab_to_idx[char] for char in input_date] + [self.input_vocab_to_idx['<EOS>']]
        output_encoded = [self.output_vocab_to_idx['<SOS>']] + [self.output_vocab_to_idx[char] for char in output_date] + [self.output_vocab_to_idx['<EOS>']]


        # Pad sequences if necessary
        input_padded = input_encoded + [self.input_vocab_to_idx['<PAD>']] * (self.max_len - len(input_encoded))
        output_padded = output_encoded + [self.output_vocab_to_idx['<PAD>']] * (self.max_len - len(output_encoded))
        
        input_padded = input_padded[:self.max_len]
        output_padded = output_padded[:self.max_len]

        return torch.tensor(input_padded, dtype=torch.long), torch.tensor(output_padded, dtype=torch.long)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # Apply embedding layer
        embedded = self.embedding(x)
        # Scale with sqrt of embed_dim
        return embedded * math.sqrt(self.embed_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=20):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
          attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # masking the padding tokens
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]

    def combine_heads(self, x):
        batch_size, _, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)
    
    def forward(self, q, k, v, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        concat_attn_output = self.combine_heads(attn_output)
        
        output = self.out_linear(concat_attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.attention(x, x, x, mask) #Self-attention
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_out = self.self_attention(x, x, x, tgt_mask)  #Self-attention
        x = self.norm1(x + self.dropout(attn_out))

        cross_attn_out = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feedforward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
                                       for _ in range(num_layers)])
    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
                                       for _ in range(num_layers)])
    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.dec_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len):
        super(Transformer, self).__init__()

        self.input_embedding = Embedding(input_vocab_size, embed_dim)
        self.output_embedding = Embedding(output_vocab_size, embed_dim)

        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.encoder = Encoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.fc_output = nn.Linear(embed_dim, output_vocab_size)
        self.max_seq_len = max_seq_len

    def generate_mask(self, src, tgt, pad_idx):
      src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
      tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
      seq_len = tgt.shape[1]
      nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
      tgt_mask = tgt_mask & nopeak_mask
      return src_mask, tgt_mask

    def forward(self, src, tgt):
        pad_idx = input_vocab_to_idx['<PAD>']
        src_mask, tgt_mask = self.generate_mask(src, tgt[:, :-1], pad_idx)
        
        src_embedded = self.pos_encoding(self.input_embedding(src)) #Apply embeddings and positional encoding to the input and ouput
        tgt_embedded = self.pos_encoding(self.output_embedding(tgt[:, :-1]))

        enc_output = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        output = self.fc_output(dec_output)

        return output

#Generate dataset
num_samples = 10000
date_pairs = generate_date_pairs(num_samples)
(input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab) = create_vocab(date_pairs)
train_pairs, test_val_pairs = train_test_split(date_pairs, test_size=0.2, random_state=42)
val_pairs, test_pairs = train_test_split(test_val_pairs, test_size=0.5, random_state=42)


# Create the datasets using the custom class
max_len = 20 # Define the max sequence length

train_dataset = DateDataset(train_pairs, input_vocab_to_idx, output_vocab_to_idx, max_len=max_len)
val_dataset = DateDataset(val_pairs, input_vocab_to_idx, output_vocab_to_idx, max_len=max_len)
test_dataset = DateDataset(test_pairs, input_vocab_to_idx, output_vocab_to_idx, max_len=max_len)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_vocab_size = len(input_vocab_to_idx)
output_vocab_size = len(output_vocab_to_idx)


print("Input Vocab size:", input_vocab_size)
print("Output Vocab size:", output_vocab_size)
print("Number of Training examples:",len(train_dataset))
print("Number of Validation examples:",len(val_dataset))
print("Number of Test examples:",len(test_dataset))
print("Input example with padding:", train_dataset[0][0])
print("Output example with padding:", train_dataset[0][1])

# Initialize Hyperparameters
embed_dim = 128
num_heads = 8
hidden_dim = 512
num_layers = 4
dropout = 0.1
learning_rate = 0.0001
num_epochs = 20

# Model Definition
model = Transformer(input_vocab_size, output_vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout, max_len)
# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=output_vocab_to_idx['<PAD>'])

# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define path to save weights
MODEL_PATH = 'date_transformer_model.pth'

# Training Loop
def train():
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        total_loss = 0
        for src, tgt in train_loader:
          src = src.to(device)
          tgt = tgt.to(device)

          optimizer.zero_grad()
          output = model(src, tgt)
          # Reshape output to be [batch*seq_len, output_vocab_size] and tgt to [batch*seq_len]
          loss = criterion(output.view(-1, output.size(-1)), tgt[:,1:].reshape(-1))
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        model.eval() # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                output = model(src, tgt)
                val_loss = criterion(output.view(-1, output.size(-1)), tgt[:,1:].reshape(-1))
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss/len(val_loader)
            print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Model saved at epoch {epoch+1}")



# Run training loop
train()

def translate_date(model, input_date, input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab, device, max_len=20):
    model.eval()
    with torch.no_grad():
        # Encode the input date
        input_encoded = [input_vocab_to_idx['<SOS>']] + [input_vocab_to_idx[char] for char in input_date] + [input_vocab_to_idx['<EOS>']]
        input_padded = input_encoded + [input_vocab_to_idx['<PAD>']] * (max_len - len(input_encoded))
        input_padded = torch.tensor(input_padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)

        # Create a start token for the decoder
        output_encoded = [output_vocab_to_idx['<SOS>']]
        output_tensor = torch.tensor(output_encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_len-1):
            # Generate the output from the model, make prediction
            prediction = model(input_padded, output_tensor)
            # Prediction is [batch, seq_len, vocab_size], we want the last element
            prediction = prediction[:, -1, :].argmax(dim=-1)
            output_encoded.append(prediction.item())
            output_tensor = torch.tensor(output_encoded, dtype=torch.long).unsqueeze(0).to(device)
            if prediction.item() == output_vocab_to_idx['<EOS>']:
              break

        # Decode the output sequence
        output_string = "".join([output_idx_to_vocab[idx] for idx in output_encoded[1:] if idx != output_vocab_to_idx['<EOS>'] and idx != output_vocab_to_idx['<PAD>']])
        return output_string


def evaluate(model, test_loader, input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab, device, max_len=20):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            #Decode the original tgt strings from encoded tensor
            output_tgt = ["".join([output_idx_to_vocab[idx] for idx in row if idx != output_vocab_to_idx['<EOS>'] and idx != output_vocab_to_idx['<PAD>']]) for row in tgt]
            for i in range(len(src)):
                input_string = "".join([input_idx_to_vocab[idx] for idx in src[i] if idx != input_vocab_to_idx['<EOS>'] and idx != input_vocab_to_idx['<PAD>'] and idx != input_vocab_to_idx['<SOS>']])
                predicted_output = translate_date(model, input_string, input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab, device, max_len)
                if predicted_output == output_tgt[i]:
                  correct_predictions += 1
                total_predictions += 1
            
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Load the best model weights
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded trained model from disk.")
else:
    print("No saved model found, please train the model first!")

test_accuracy = evaluate(model, test_loader, input_vocab_to_idx, input_idx_to_vocab, output_vocab_to_idx, output_idx_to_vocab, device, max_len=max_len)
print(f"Test Accuracy: {test_accuracy:.2f}%")
