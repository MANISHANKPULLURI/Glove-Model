import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


def load_glove_model(model_dir):
    embeddings = np.load(f"{model_dir}/embeddings.npy")
    with open(f"{model_dir}/word_to_id.pkl", "rb") as f:
        word_to_id = pickle.load(f)
    with open(f"{model_dir}/id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)
    with open(f"{model_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return embeddings, word_to_id, id_to_word, metadata


def text_to_indices(text, word_to_id, max_seq_len=600):
    tokens = text.lower().split()
    indices = [word_to_id.get(w, 0) for w in tokens]
    if len(indices) > max_seq_len:
        indices = indices[:max_seq_len]
    return indices


def prepare_cnn_dataset(file_path, word_to_id, max_seq_len=600):
    sequences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            genres_part = parts[0]
            text_part = parts[1]
            primary_genre = genres_part.split(',')[0].strip()
            indices = text_to_indices(text_part, word_to_id, max_seq_len)
            sequences.append(torch.tensor(indices, dtype=torch.long))
            labels.append(primary_genre)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_array = np.array(labels)
    counts = Counter(labels_array)
    filtered_indices = [i for i, label in enumerate(labels_array) if counts[label] >= 3]
    return padded_sequences[filtered_indices], labels_array[filtered_indices]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None,
                 filter_sizes=[3, 4, 5], num_filters=128, dropout=0.5, max_seq_len=600):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_cnn_model(model_dir, train_dataset_path, test_dataset_path, output_pkl, max_seq_len=600,
                   filter_sizes=[3,4,5], num_filters=128, epochs=30,
                   batch_size=32, lr=0.001, dropout=0.3, weight_decay=0.0005):

    embeddings, word_to_id, id_to_word, metadata = load_glove_model(model_dir)
    vocab_size, embedding_dim = embeddings.shape

    X_train, y_train = prepare_cnn_dataset(train_dataset_path, word_to_id, max_seq_len)
    X_test, y_test = prepare_cnn_dataset(test_dataset_path, word_to_id, max_seq_len)

    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    valid_labels = sorted([label for label in set(y_train) | set(y_test)
                           if train_counts[label] >= 3 and test_counts[label] >= 3])

    label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_mask = np.array([label in valid_labels for label in y_train])
    test_mask = np.array([label in valid_labels for label in y_test])

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    y_test_idx = np.array([label_to_idx[label] for label in y_test])

    class_counts = np.bincount(y_train_idx)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weights = torch.tensor(class_weights, dtype=torch.float)

    train_dataset = TensorDataset(X_train, torch.tensor(y_train_idx, dtype=torch.long))
    test_dataset = TensorDataset(X_test, torch.tensor(y_test_idx, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=len(valid_labels),
        pretrained_embeddings=embeddings,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_state = None
    best_model_epoch = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            val_accuracy = 100. * val_correct / val_total
            val_accuracies.append(val_accuracy)
            if val_accuracy > best_test_acc:
                best_test_acc = val_accuracy
                best_model_state = model.state_dict().copy()
                best_model_epoch = epoch + 1
            model.train()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    correct_preds = sum(np.array(y_true) == np.array(y_pred))
    wrong_preds = len(y_true) - correct_preds

    print("\nFINAL REPORT:")
    print(f"Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total Predictions: {len(y_true)}")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Wrong Predictions: {wrong_preds}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[idx_to_label[i] for i in range(len(idx_to_label))], zero_division=0))

    final_model_path = output_pkl
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model_data = {
        "final_epoch": len(train_accuracies),
        "best_epoch": best_model_epoch,
        "model_state": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_classes": len(valid_labels),
        "filter_sizes": filter_sizes,
        "num_filters": num_filters,
        "max_seq_len": max_seq_len,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "best_val_accuracy": best_test_acc,
        "test_accuracy": acc,
        "test_f1": f1,
        "val_accuracies": val_accuracies,
        "train_losses_history": train_losses,
        "train_accuracies_history": train_accuracies
    }
    with open(final_model_path, "wb") as f:
        pickle.dump(model_data, f)
    return model, acc, f1


models_to_train = [
    {
        "model_dir": "wordembeddings/glove_model_30k",
        "train_dataset_path": "english_30k_mapped.txt",
        "test_dataset_path": "english_test_mapped.txt",
        "output_pkl": "models/500itrenglish30k_cnn.pkl",
        "max_seq_len": 600,
        "filter_sizes": [2, 3, 4, 5],
        "num_filters": 256,
        "epochs": 30,
        "batch_size": 64,
        "lr": 0.0005,
        "dropout": 0.5,
        "weight_decay": 0.0001
    }
]

for config in models_to_train:
    train_cnn_model(**config)
    print(f"Trained model saved to {config['output_pkl']}")