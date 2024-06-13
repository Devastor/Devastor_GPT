import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import re
import json
import os

# Простой токенайзер на основе разделения слов
def simple_tokenizer(text):
    return text.lower().split()

# Обратная функция для токенайзера
def detokenize(tokens):
    return ' '.join(tokens)

# Создание словаря на основе текстов
def build_vocab(texts):
    vocab = set()
    for text in texts:
        tokens = simple_tokenizer(text)
        vocab.update(tokens)
    return {word: idx for idx, word in enumerate(vocab)}

# Датасет
class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = simple_tokenizer(text)
        input_ids = [self.vocab.get(token, 0) for token in tokens]
        input_ids = input_ids + [0] * (self.max_len - len(input_ids))  # Padding
        input_ids = input_ids[:self.max_len]
        return torch.tensor(input_ids), torch.tensor([1] * len(tokens) + [0] * (self.max_len - len(tokens)))

# Класс механизма внимания
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32, device=q.device))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Правильное расширение маски
            scores += (mask * -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        
        return output

# Класс слоя трансформера
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2

# Полная модель GPT
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dim_feedforward, max_seq_len, dropout=0.1):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, x, mask=None):
        if x.size(1) != self.max_seq_len:
            padding = torch.zeros(x.size(0), self.max_seq_len - x.size(1), dtype=torch.long, device=x.device)
            x = torch.cat((x, padding), dim=1)
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        if x.max() >= self.token_embedding.num_embeddings:
            raise ValueError(f"Token index out of range: {x.max().item()} >= {self.token_embedding.num_embeddings}")
        
        x = self.token_embedding(x) + self.position_embedding(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.fc_out(x)
        return logits

# Функция обучения
def train(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    print("Начало обучения...", flush=True)
    total_start_time = time.time()  # Запуск общего таймера
    for epoch in range(num_epochs):
        print(f'Эпоха {epoch+1} началась...', flush=True)
        epoch_start_time = time.time()  # Запуск таймера эпохи
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()  # Запуск таймера батча
            try:
                inputs, attention_mask = batch
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                optimizer.zero_grad()
                try:
                    outputs = model(inputs, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print("ERROR in forward/backward pass: ", e, flush=True)
                    continue  # Пропустить текущий батч в случае ошибки
                batch_time = time.time() - batch_start_time  # Вычисление времени выполнения батча
                print(f'Эпоха {epoch+1}/{num_epochs}, Батч {batch_idx}, Потери: {loss.item()}, Время: {batch_time:.2f} сек.', flush=True)
            except Exception as e:
                print("ERROR in batch processing: ", e, flush=True)
                continue  # Пропустить текущий батч в случае ошибки
        epoch_time = time.time() - epoch_start_time  # Вычисление времени выполнения эпохи
        print(f'Эпоха {epoch+1}/{num_epochs} завершена, Время эпохи: {epoch_time:.2f} сек.', flush=True)
    total_time = time.time() - total_start_time  # Вычисление общего времени выполнения
    print(f'Обучение завершено, Общее время: {total_time:.2f} сек.', flush=True)

# Функция загрузки текстов из файла
def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts]

# Основная функция для генерации текста
def generate_text(model, vocab, prompt, max_length, device):
    model.eval()
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    tokens = simple_tokenizer(prompt)
    input_ids = torch.tensor([vocab.get(token, 0) for token in tokens]).unsqueeze(0).to(device)

    generated_tokens = tokens

    for _ in range(max_length - len(tokens)):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = reverse_vocab.get(next_token_id, '<unk>')
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)

    return detokenize(generated_tokens)

# Основная функция программы
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 5      # Количество эпох обучения, можно увеличить
    d_model = 512  # Размерность модели
    num_layers = 12  # Количество слоев трансформера
    num_heads = 8  # Количество головок внимания
    dim_feedforward = 2048  # Размерность внутреннего слоя
    max_seq_len = 1024  # Максимальная длина последовательности
    dropout = 0.1  # Вероятность дропаута

    while True:
        print("Выберите режим:")
        print("1. Режим обучения")
        print("2. Режим генерации")
        print("3. Выход")
        
        choice = input("Введите номер режима: ")

        if choice == "1":
            print("Режим обучения")
            # Загрузка текстов из файла
            texts = load_texts("train_text.txt")
            print(f"Количество текстов: {len(texts)}", flush=True)

            # Создание словаря и датасета
            vocab = build_vocab(texts)
            vocab_size = len(vocab)

            model = GPTModel(vocab_size, d_model, num_layers, num_heads, dim_feedforward, max_seq_len, dropout).to(device)
            # Подсчет количества параметров
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Количество параметров в модели: {total_params}", flush=True)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

            dataset = TextDataset(texts, vocab, max_seq_len)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            print("Инициализация обучения...", flush=True)
            train(model, dataloader, criterion, optimizer, num_epochs=epochs, device=device)

            torch.save(model.state_dict(), "gpt_model_S.pth")
            with open("vocab.json", "w") as f:
                json.dump(vocab, f)
            print("Модель и словарь сохранены", flush=True)
            print(model)

        elif choice == "2":
            print("Режим генерации")
            if not os.path.exists("gpt_model_S.pth") or not os.path.exists("vocab.json"):
                print("Сначала обучите модель в режиме обучения", flush=True)
                continue

            with open("vocab.json", "r") as f:
                vocab = json.load(f)
            vocab_size = len(vocab)

            model = GPTModel(vocab_size, d_model, num_layers, num_heads, dim_feedforward, max_seq_len, dropout).to(device)
            model.load_state_dict(torch.load("gpt_model_S.pth", map_location=device))

            while True:
                prompt = input("Введите начальный текст (или 'exit' для выхода): ")
                if prompt.lower() == "exit":
                    break
                generated_text = generate_text(model, vocab, prompt, max_length=max_seq_len, device=device)
                print(f"Сгенерированный текст: {generated_text}")

        elif choice == "3":
            print("Выход")
            break
        else:
            print("Неверный выбор, попробуйте снова.")

if __name__ == "__main__":
    main()
