import pandas as pd
import re


def dataset_cleaner(path_to_data, path_to_save, max_rows=None):
    # Читаем исходный файл .txt
    with open(path_to_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Если указано ограничение по числу строк — берём первые max_rows
    if max_rows is not None and max_rows > 0:
        lines = lines[:max_rows]

    # Создаём DataFrame, убираем пустые строки
    df = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})

    def clean_text(text):
        text = text.lower()  # нижний регистр
        
        # Удаляем ссылки
        text = re.sub(r"http\S+|www\S+|https\S+", "<URL>", text)
        # Удаляем упоминания пользователей @username
        text = re.sub(r"@\w+", "<USER>", text)
        # Заменяем хештеги #hashtag на специальный токен или убираем #
        text = re.sub(r"#", "", text)
        # Удаляем все символы кроме букв, цифр и пробелов
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        # Убираем повторяющиеся пробелы
        text = re.sub(r"\s+", " ", text).strip()
        # Сокращаем повторяющиеся символы (аааа → а)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        return text

    df['clean_text'] = df['text'].apply(clean_text)
    df[['clean_text']].to_csv(path_to_save, index=False)


if __name__ == "__main__":
    dataset_cleaner("data/tweets.txt", "data/tweets_cleaned_small.csv", 500000)
    