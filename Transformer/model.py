import torch
import torch.nn as nn

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(16, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device)
        out = self.token_embedding(x) + self.pos_embedding(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        for layer in self.layers:
            out = layer(out, out, tgt_mask=mask, memory_mask=mask)
        logits = self.fc_out(out)
        return logits