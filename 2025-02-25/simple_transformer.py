import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        # x shape: [batch_size, seq_length, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections and reshape for multi-head attention
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)

        # Reshape and project back to d_model
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(out)

        return out, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention with residual connection and layer norm
        attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final linear layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        # Embed source tokens and add positional encoding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        return src

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        # Embed target tokens and add positional encoding
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)

        return tgt

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        memory = self.encode(src, src_mask)

        # Decode
        output = self.decode(tgt, memory, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.fc(output)

        return output


# Example usage
if __name__ == "__main__":
    # Small toy example
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    batch_size = 2
    seq_length = 10

    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size)

    # Random source and target sequences
    src = torch.randint(1, src_vocab_size, (batch_size, seq_length))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_length))

    # Create masks (in a real scenario, these would mask padding tokens)
    src_mask = torch.ones(batch_size, 1, seq_length)
    # Create causal mask for decoder (cannot look at future tokens)
    tgt_mask = torch.tril(torch.ones(batch_size, seq_length, seq_length))

    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)

    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer model successfully created and executed!")