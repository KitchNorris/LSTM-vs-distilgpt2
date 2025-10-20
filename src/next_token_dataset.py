import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class PredictDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, seq_len=2):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=max_length, truncation=True)
            if len(token_ids) < seq_len: continue
            
            context = token_ids[:-1]
            target = token_ids[1:]
            self.samples.append((context, target))
                
    def __len__(self):
        return len(self.samples) # верните размер датасета

    def __getitem__(self, idx):
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)
    

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_padded, ys_padded

def choose_max_len(texts, percentile=90):
    lengths = [len(text.split()) for text in texts]
    return int(np.percentile(lengths, percentile))


def get_datasets(tokenizer, path_to_csv):
    #df_cleaned = pd.read_csv('data/tweets_cleaned.csv')
    df_cleaned = pd.read_csv(path_to_csv)

    max_len = choose_max_len(df_cleaned["clean_text"].dropna().astype(str).tolist())
    print('Характеристики датасета:')
    print(f'Максимальная последовательность токенов (персентиль=90): {max_len}')

    # Удаляем строки с NaN и некорректными значениями в clean_text
    df_cleaned = df_cleaned.dropna(subset=["clean_text"])
    df_cleaned = df_cleaned[df_cleaned["clean_text"].apply(lambda x: isinstance(x, str))]
    df_cleaned = df_cleaned[df_cleaned["clean_text"].str.strip() != ""]
    
    # Делим выборку корректно
    train, all_test = train_test_split(df_cleaned, test_size=0.20, random_state=42)
    val, test = train_test_split(all_test, test_size=0.50, random_state=42)
    print(f'train size={len(train)}, val size={len(val)}, test size={len(test)}')

    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    train_dataset = PredictDataset(train["clean_text"].tolist(), tokenizer, max_len)
    val_dataset = PredictDataset(val["clean_text"].tolist(), tokenizer, max_len)
    #test_dataset = PredictDataset(test["clean_text"].tolist(), tokenizer, max_len)

    # Принтим пару примеров токенов
    for i in range(3):
        x, y = train_dataset[i]
        print(f"Пример {i} - X токены: {x.tolist()}")
        print(f"Пример {i} - Y токены: {y.tolist()}")
        print(f"Пример {i} - X текст: {tokenizer.decode(x.tolist())}")
        print(f"Пример {i} - Y текст: {tokenizer.decode(y.tolist())}")

    return train_dataset, val_dataset #, test_dataset


if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    get_datasets(tokenizer, 'data/tweets_cleaned_small.csv')
