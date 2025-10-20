import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import evaluate


# ==============================
# --- Вспомогательные функции ---
# ==============================

rouge = evaluate.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_text(token_ids, tokenizer):
    """Декодирует список идентификаторов токенов в строку"""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_one_token(model, input_ids, hidden=None):
    """Генерирует один следующий токен (используется для демонстрации)"""
    model.eval()
    with torch.no_grad():
        logits, hidden = model(input_ids, hidden)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token, hidden


def generate_completion(model, input_ids, max_len=50):
    """Генерация завершения последовательности"""
    model.eval()
    generated = input_ids.clone()
    hidden = None
    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(generated[:, -1:].to(device), hidden)
            probs = torch.softmax(logits[:, -1], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def calculate_val_loss(model, val_loader, criterion):
    """Вычисление среднего loss на валидации"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _ = model(x_batch)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def evaluate_rouge(model, dataloader, tokenizer):
    """Вычисление ROUGE метрик"""
    model.eval()
    predictions, references = [], []
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        seq_len = x_batch.shape[1]
        split_idx = (3 * seq_len) // 4
        input_ids = x_batch[:, :split_idx]
        target_ids = x_batch[:, split_idx:]

        generated_ids = generate_completion(model, input_ids, max_len=target_ids.shape[1])
        pred_texts = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(target_ids.cpu(), skip_special_tokens=True)

        predictions.extend(pred_texts)
        references.extend(ref_texts)

    results = rouge.compute(predictions=predictions, references=references)
    return results


# ==============================
# --- Основной цикл обучения ---
# ==============================

def training_loop(model, tokenizer, train_loader, val_loader, n_epochs=10, save_path="models/model_weights.pth"):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    model.to(device)

    # списки для последующей отрисовки графиков
    train_losses, val_losses, rouge_scores = [], [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs, _ = model(x_batch)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # средний train_loss
        train_loss /= len(train_loader)

        # вычисляем val_loss и ROUGE
        val_loss = calculate_val_loss(model, val_loader, criterion)
        rouge_result = evaluate_rouge(model, val_loader, tokenizer)

        # сохраняем в списки
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        rouge_scores.append(rouge_result['rouge1'])

        print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        print(f"Epoch {epoch+1}: ROUGE: {rouge_result}")

        # сохраняем веса
        torch.save(model.state_dict(), save_path)

        # --- Пример генерации после эпохи ---
        model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_loader):
                input_ids = x_batch[:, :x_batch.size(1) * 3 // 4].to(device)

                # Используем generate_one_token для демонстрации
                next_token, _ = generate_one_token(model, input_ids)
                generated_ids = generate_completion(model, input_ids, max_len=x_batch.size(1)//4)

                print("\n--- Пример генерации ---")
                input_text = tokens_to_text(input_ids[0].cpu().tolist(), tokenizer)
                pred_word = tokens_to_text([next_token[0].item()], tokenizer)
                generated_text = input_text + " " + pred_word

                print("Вход:", input_text)
                print("Референс:", tokens_to_text(y_batch[0].cpu().tolist(), tokenizer))
                print("Генерация одного слова:", generated_text)
                print("Генерация последовательности:", tokens_to_text(generated_ids[0].cpu().tolist(), tokenizer))
                break  # показываем только один пример после каждой эпохи

    print("\n=== Обучение завершено ===")
    return train_losses, val_losses, rouge_scores
