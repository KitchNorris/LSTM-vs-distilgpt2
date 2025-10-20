import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCausalAttentionPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=3, num_heads=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )

        # multi-head attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len)
        hidden: (h, c)
        returns: logits (batch, seq_len, vocab), hidden
        """
        emb = self.embedding(x)
        lstm_out, hidden = self.lstm(emb, hidden)  # (B, L, H)

        seq_len = lstm_out.size(1)

        # --- создаём causal mask ---
        # Верхний треугольник (включая диагональ) = False (видим прошлое и себя)
        # Нижний треугольник = True (запрещаем смотреть вперёд)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)
        out = self.layer_norm(lstm_out + self.dropout(attn_out))

        logits = self.fc(out)
        return logits, hidden

    def generate(self, start_tokens, max_len=20, temperature=1.0):
        """
        Авторегрессивная генерация по одному токену за шаг
        """
        self.eval()
        generated = start_tokens
        hidden = None

        for _ in range(max_len):
            # подаём только последний токен (а не всю последовательность!)
            logits, hidden = self.forward(generated[:, -1:], hidden)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
