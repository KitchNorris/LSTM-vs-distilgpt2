# --- Зависимости ---
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm

# --- Настройка устройства и ROUGE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rouge = evaluate.load("rouge")


# ------------------------------
#  Функции для трансформера
# ------------------------------

def prepare_gpt_tokenizer(model_name="distilgpt2"):
    """
    Загружает tokenizer и корректирует pad_token (GPT2 не имеет pad_token по умолчанию).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Если pad_token отсутствует, используем eos_token как pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gpt_model(model_name="distilgpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def generate_continuations(model,
                           tokenizer,
                           input_ids,
                           target_len,
                           max_new_tokens=None,
                           temperature=0.8,
                           top_k=50,
                           top_p=0.95,
                           do_sample=True,
                           num_return_sequences=1):
    """
    Бэтч-генерация продолжений для input_ids.
    - input_ids: tensor (batch, prefix_len) на device
    - target_len: сколько токенов генерировать (целая последняя 1/4)
    Возвращает tensor (batch, prefix_len + target_len) — полная последовательность.
    """
    # model.generate принимает max_new_tokens (новых токенов)
    if max_new_tokens is None:
        max_new_tokens = target_len

    # attention_mask: 1 для непаддед токенов
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
        # use_cache=True by default (speeds up autoreg)
    )
    return generated  # (batch * num_return_sequences, prefix_len + new_tokens)


# ------------------------------
#  Оценка ROUGE с трансформером
# ------------------------------

def evaluate_transformer_rouge(model, tokenizer, dataloader,
                               sample_limit=None,
                               generation_kwargs=None,
                               verbose_examples=2):
    """
    Проходит по даталоадеру, для каждого примера:
      - отрезает первые 3/4 -> feed в модель как контекст
      - генерирует последние 1/4 токенов
      - собирает predictions и references (как строки) и считает ROUGE
    Параметры:
      - sample_limit: если нужно ограничить число батчей (для быстрого теста)
      - generation_kwargs: dict с параметрами для generate_continuations
      - verbose_examples: сколько примеров распечатать
    Возвращает: results (словарь ROUGE) и пару списков (preds, refs)
    """
    model.eval()
    if generation_kwargs is None:
        generation_kwargs = dict(temperature=0.8, top_k=50, top_p=0.95, do_sample=True)

    predictions = []
    references = []
    printed = 0
    processed_batches = 0

    for x_batch, y_batch in tqdm(dataloader, desc="Eval transformer ROUGE"):
        # x_batch: (batch, seq_len) -- уже числовые токен-иденты
        batch_size, seq_len = x_batch.shape
        split_idx = (3 * seq_len) // 4
        prefix_len = split_idx
        target_len = seq_len - split_idx

        # Готовим input_ids (prefix) на устройстве
        input_ids = x_batch[:, :split_idx].to(device)

        # Генерируем (батчово)
        with torch.no_grad():
            gen_full = generate_continuations(
                model, tokenizer, input_ids,
                target_len,
                max_new_tokens=target_len,
                **generation_kwargs
            )  # shape (batch, prefix_len + target_len)

        # Получаем сгенерированную часть: последние target_len токенов
        gen_continuations = gen_full[:, -target_len:].cpu()
        target_ids = x_batch[:, split_idx:].cpu()  # ожидаемая часть

        # --- Дополнительно генерируем только 1 слово ---
        with torch.no_grad():
            gen_one = generate_continuations(
                model, tokenizer, input_ids,
                target_len=1,
                max_new_tokens=1,
                **generation_kwargs
            )
        one_word_preds = tokenizer.batch_decode(gen_one[:, -1:], skip_special_tokens=True)

        # Декодируем в строки
        pred_texts = tokenizer.batch_decode(gen_continuations, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        predictions.extend(pred_texts)
        references.extend(ref_texts)

        # Печатаем несколько примеров для визуальной проверки
        if printed < verbose_examples:
            for i in range(min(verbose_examples - printed, batch_size)):
                idx = i
                prefix_text = tokenizer.decode(input_ids[idx].cpu().tolist(), skip_special_tokens=True)
                pred_cont = pred_texts[idx]
                ref_cont = ref_texts[idx]
                print("\n--- Пример (Transformer) ---")
                print("Input prefix:", prefix_text)
                print("Predicted single word:", one_word_preds[idx])
                print("Predicted continuation:", pred_cont)
                print("Reference continuation:", ref_cont)
                printed += 1

        processed_batches += 1
        if sample_limit is not None and processed_batches >= sample_limit:
            break

    # Вычисляем ROUGE
    results = rouge.compute(predictions=predictions, references=references)
    print("\nROUGE scores:", results)
    return results, predictions, references

