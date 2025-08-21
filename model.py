import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class PianoTransformer(nn.Module):
    def __init__(self, input_dim, num_tokens, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_tokens)

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        out = self.transformer_encoder(src)
        out = out.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.output_proj(out)  # (batch, seq_len, num_tokens)
        return logits
