import torch
import torch.nn as nn
from tqdm import tqdm
import evaluate


# Создаем объект ROUGE
rouge = evaluate.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda") 

def tokens_to_text(token_ids, tokenizer):
    # декодируем список идентификаторов токенов в строку
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def generate_one_token(model, input_ids, hidden=None):
    model.eval()
    with torch.no_grad():
        logits, hidden = model(input_ids, hidden)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token, hidden

def generate_completion(model, input_ids, max_len=50):
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

def evaluate_rouge1(model, dataloader, tokenizer):
    model.eval()
    predictions = []
    references = []
    for x_batch, y_batch in dataloader:
        batch_size, seq_len = x_batch.shape
        # Отрезаем 3/4 последовательности для начала
        split_idx = (3 * seq_len) // 4
        input_ids = x_batch[:, :split_idx].to(device)
        target_ids = x_batch[:, split_idx:].to(device)  # target - оставшиеся 1/4

        generated_ids = generate_completion(model, input_ids, max_len=target_ids.shape[1])

        for pred, ref in zip(generated_ids, y_batch):
            pred_text = tokens_to_text(pred.cpu().tolist(), tokenizer)
            ref_text = tokens_to_text(ref.cpu().tolist(), tokenizer)
            predictions.append(pred_text)
            references.append(ref_text)
    
    results = rouge.compute(predictions=predictions, references=references)
    return results

def evaluate_rouge(model, dataloader, tokenizer):
    model.eval()
    predictions = []
    references = []
    for x_batch, y_batch in dataloader:
        batch_size, seq_len = x_batch.shape
        split_idx = (3 * seq_len) // 4
        input_ids = x_batch[:, :split_idx].to(device)
        target_ids = x_batch[:, split_idx:].to(device)

        generated_ids = generate_completion(model, input_ids, max_len=target_ids.shape[1])

        pred_texts = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(y_batch.cpu(), skip_special_tokens=True)

        predictions.extend(pred_texts)
        references.extend(ref_texts)

    results = rouge.compute(predictions=predictions, references=references)
    return results


def training_loop(model, tokenizer, train_loader, val_loader, n_epochs=10, save_path="models/model_weights.pth"):
    # Функция потерь и оптимизатор
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    model.to(device)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs, _ = model(x_batch)
            # Reshape для loss: (batch * seq_len, vocab_size)
            outputs = outputs.view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

        # Вычисляем ROUGE на валидации
        rouge_scores = evaluate_rouge(model, val_loader, tokenizer)
        print(f"Epoch {epoch+1}: ROUGE: {rouge_scores}")

        # Сохраняем веса
        torch.save(model.state_dict(), save_path)

        # Показываем примеры автодополнений из валидации
        model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_loader):
                input_ids = x_batch[:, :x_batch.size(1)*3//4].to(device)
                generated_ids = generate_completion(model, input_ids, max_len=x_batch.size(1)//4)
                print("Input:", tokens_to_text(input_ids[0].cpu().tolist(), tokenizer))
                print("Generated:", tokens_to_text(generated_ids[0].cpu().tolist(), tokenizer))
                print("Reference:", tokens_to_text(y_batch[0].cpu().tolist(), tokenizer))
                break  # показываем только один пример после эпохи
