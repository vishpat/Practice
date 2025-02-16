import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import os
import tiktoken # Import tiktoken

# Set up the environment by setting a random seed for reproducibility and choosing the device (GPU if available)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Flag to control saving and loading of models
SAVE_MODEL = True
LOAD_MODEL = False
MODEL_PATH = "toy_transformer_model_1.pth"

# Training sentences, each with exactly 4 words, to train the model
training_sentences = [
    "The sun rises slowly.",
    "The sun sets quietly.",
    "The fire crackles loudly.",
    "The fire burns steadily.",
    "The bird sings sweetly.",
    "The bird flies high.",
    "The bell tolls loudly.",
    "The bell rings softly.",
    "The wind blows gently.",
    "The wind howls fiercely.",
    "The river flows smoothly.",
    "The river rushes swiftly."
]

# Fine-tuning sentences, also with 4 words, to refine the model’s understanding of these specific patterns
fine_tuning_sentences = [
    "The sun sets quietly.",
    "The fire burns brightly.",
    "The bird sings sweetly.",
    "The bell rings softly.",
    "The wind blows gently.",
    "The river flows smoothly."
]

# Initialize tiktoken tokenizer with 'gpt2' encoding
tokenizer = tiktoken.get_encoding("gpt2")

# Create vocabulary from tiktoken (we won't use our custom vocab anymore)
# For tiktoken, vocabulary is implicitly handled by the tokenizer
vocab_size = tokenizer.n_vocab # Get vocab size from tokenizer


def encode_sentences_tiktoken(sentences, tokenizer):
    """
    Encode sentences into tensors of token indices using tiktoken tokenizer.

    Args:
    sentences (list of str): List of sentences to encode.
    tokenizer (tiktoken.Encoding): tiktoken tokenizer.

    Returns:
    encoded (list of torch.Tensor): List of encoded sentences as tensors of token indices.
    """
    encoded = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence) # Encode using tiktoken
        encoded.append(torch.tensor(tokens))
    return encoded

# Encode the training and fine-tuning sentences into tensors using tiktoken
input_tensors = encode_sentences_tiktoken(training_sentences, tokenizer)
target_tensors = encode_sentences_tiktoken(training_sentences, tokenizer)
fine_tuning_tensors = encode_sentences_tiktoken(fine_tuning_sentences, tokenizer)


# Model configuration parameters (vocab_size is now from tiktoken)
d_model = 256  # Dimensionality of the model (size of the embedding vectors)
d_ff = 1024    # Dimensionality of the feed-forward layers
num_heads = 8  # Number of attention heads in the multi-head attention mechanism
num_layers = 6 # Number of layers in the encoder and decoder
dropout_rate = 0.3  # Dropout rate for regularization


# Positional Encoding layer to add positional information to the input embeddings (No changes needed)
# Simplified Learned Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        positional_embeddings = self.embedding(positions)
        return x + positional_embeddings

# Multi-Head Attention layer (No changes needed)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)

        context_layer = torch.matmul(attention_weights, v)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(context_layer)
        return output

# Feed-Forward Network layer (No changes needed)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Add & Norm layer (No changes needed)
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

# Encoder Block (No changes needed)
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x

# Decoder Block (No changes needed)
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.add_norm1(x, self_attn_output)
        enc_dec_attn_output = self.enc_dec_attention(enc_output, enc_output, x, src_mask)
        x = self.add_norm2(x, enc_dec_attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)
        return x

# Transformer model (vocab_size is now from tiktoken)
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        for layer in self.encoder:
            src = layer(src, src_mask)
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        output = self.fc_out(tgt)
        return output

# Generate square subsequent mask (No changes needed)
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Training function (Loss needs to be adjusted to vocab_size)
def train_transformer(model, input_tensors, target_tensors, fine_tuning_tensors=None, num_epochs=500):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (src, tgt) in enumerate(zip(input_tensors, target_tensors)):
            src, tgt = src.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            output = model(pos_encoder(embedding(src)), pos_encoder(embedding(tgt_input)), tgt_mask=tgt_mask)

            output = output.view(-1, vocab_size) # Use vocab_size from tiktoken
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if fine_tuning_tensors and epoch == num_epochs // 2:
            print(f"Starting fine-tuning at epoch {epoch}")
            input_tensors = fine_tuning_tensors
            target_tensors = fine_tuning_tensors

        scheduler.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(input_tensors):.4f}')

# Save and Load model functions (No changes needed)
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

# Initialize positional encoding and embedding layers (embedding needs vocab_size from tiktoken)
pos_encoder = PositionalEncoding(d_model).to(device)
embedding = nn.Embedding(vocab_size, d_model).to(device) # Use vocab_size from tiktoken

# Initialize the transformer model (vocab_size is now from tiktoken)
model = Transformer(d_model, num_heads, d_ff, num_layers, vocab_size).to(device) # Use vocab_size from tiktoken

if LOAD_MODEL and os.path.exists(MODEL_PATH):
    print("Loading model from checkpoint...")
    load_model(model, MODEL_PATH)
else:
    print("Training the model...")
    train_transformer(model, input_tensors, target_tensors, fine_tuning_tensors, num_epochs=500)

    if SAVE_MODEL:
        print("Saving model checkpoint...")
        save_model(model, MODEL_PATH)

# Prediction function with tiktoken tokenizer
def predict_two_words_beam_search_full_context_dynamic(model, prompt, beam_width=5, repetition_penalty=5.0):
    model.eval()
    prompt_tokens = tokenizer.encode(prompt.lower()) # Encode prompt using tiktoken

    src = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    beams = [(prompt_tokens, 0)]  # (token sequence, score)

    for _ in range(2):  # Predict exactly two words
        new_beams = []
        for tokens, score in beams:
            tgt = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            tgt_encoded = pos_encoder(embedding(tgt).float())

            src_encoded = pos_encoder(embedding(src).float())
            tgt_mask = generate_square_subsequent_mask(tgt_encoded.size(1)).to(device)

            output = model(src_encoded, tgt_encoded, tgt_mask=tgt_mask)
            logits = output[:, -1, :]

            for token_id in set(tokens):
                logits[:, token_id] /= repetition_penalty

            top_k_probs, top_k_indices = torch.topk(F.softmax(logits, dim=-1), beam_width)

            for i in range(beam_width):
                new_score = score + torch.log(top_k_probs[0][i])
                new_tokens = tokens + [top_k_indices[0][i].item()]
                new_beams.append((new_tokens, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    best_tokens = beams[0][0]

    return tokenizer.decode(best_tokens) # Decode using tiktoken


# Example inference with the updated beam search function
predicted_sentence_1 = predict_two_words_beam_search_full_context_dynamic(model, "The sun")
print("Predicted Sentence #1:", predicted_sentence_1)

predicted_sentence_2 = predict_two_words_beam_search_full_context_dynamic(model, "The fire")
print("Predicted Sentence #2:", predicted_sentence_2)

predicted_sentence_3 = predict_two_words_beam_search_full_context_dynamic(model, "The bird")
print("Predicted Sentence #3:", predicted_sentence_3)

predicted_sentence_4 = predict_two_words_beam_search_full_context_dynamic(model, "The wind")
print("Predicted Sentence #4:", predicted_sentence_4)

predicted_sentence_5 = predict_two_words_beam_search_full_context_dynamic(model, "The river")
print("Predicted Sentence #5:", predicted_sentence_5)