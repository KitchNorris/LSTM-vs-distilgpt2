import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTokenPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        """
        x: LongTensor shape (batch_size, seq_len) входная последовательность токенов
        hidden: (h0, c0) скрытые состояния LSTM
        возвращает: выходные логиты (batch_size, seq_len, vocab_size), последние скрытые состояния
        """
        emb = self.embedding(x)               # (batch_size, seq_len, emb_dim)
        out, hidden = self.lstm(emb, hidden)  # out: (batch_size, seq_len, hidden_dim)
        logits = self.fc(out)                 # (batch_size, seq_len, vocab_size)
        return logits, hidden

    def generate(self, start_tokens, max_len=10, temperature=1.0):
        """
        Генерирует последовательность, начиная с start_tokens.
        start_tokens: LongTensor shape (batch_size, seq_len)
        max_len: макс длина генерируемой последовательности
        """
        self.eval()
        generated = start_tokens
        hidden = None

        for _ in range(max_len):
            logits, hidden = self.forward(generated[:, -1:].long(), hidden)  # прогноз только по последнему токену
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            #probs = F.softmax(logits[:, -1, :], dim=-1)                      # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)             # сэмплируем следующий токен
            generated = torch.cat([generated, next_token], dim=1)            # добавляем к последовательности

        return generated
