import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

def load_preprocessed_data(data_dir):
    print(f"Loading preprocessed data from {data_dir}...")
    with open(f"{data_dir}/cooccurrence_data.pkl", "rb") as f:
        co_occurrence_data = pickle.load(f)
    with open(f"{data_dir}/word_to_id.pkl", "rb") as f:
        word_to_id = pickle.load(f)
    with open(f"{data_dir}/id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)
    with open(f"{data_dir}/preprocessing_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded:")
    print(f"  - Vocabulary: {metadata['vocab_size']} words")
    print(f"  - Co-occurrences: {metadata['non_zero_entries']} entries")
    print(f"  - Window size: {metadata['window_size']}")
    print(f"  - Dataset: {metadata['dataset']}")
    return co_occurrence_data, word_to_id, id_to_word, metadata



def cpu_batch(args):
    indices_i, indices_j, values, word_vec, context_vec, b, c, learning_rate, x_max, alpha = args

    Xij = values
    wgt = np.where(Xij < x_max, (Xij/x_max) ** alpha, 1.0)

    word_vecs_batch = word_vec[indices_i]
    context_vecs_batch = context_vec[indices_j]
    b_batch = b[indices_i]
    c_batch = c[indices_j]

    dots = np.sum(word_vecs_batch * context_vecs_batch, axis=1)
    diff = dots + b_batch + c_batch - np.log(Xij + 1)
    batch_loss = np.sum(wgt * diff ** 2)

    grad = 2 * wgt * diff
    grad_reshaped = grad.reshape(-1, 1)
    word_vec_updates = grad_reshaped * context_vecs_batch
    context_vec_updates = grad_reshaped * word_vecs_batch

    for idx, update in zip(indices_i, word_vec_updates):
        word_vec[idx] -= learning_rate * update
    for idx, update in zip(indices_j, context_vec_updates):
        context_vec[idx] -= learning_rate * update
    for idx, update in zip(indices_i, grad):
        b[idx] -= learning_rate * update
    for idx, update in zip(indices_j, grad):
        c[idx] -= learning_rate * update

    return float(batch_loss)



def train_glove(data_dir, dimension=100, epochs=100, learning_rate=0.01,
                x_max=100, alpha=0.75, batch_size=30000, num_workers=6):
    co_occurrence_data, word_to_id, id_to_word, metadata = load_preprocessed_data(data_dir)

    vocab_size = metadata['vocab_size']
    non_zero_indices = co_occurrence_data['indices']
    non_zero_values = co_occurrence_data['values']

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")


    word_vec = torch.randn(vocab_size, dimension, dtype=torch.float32, device=device) * 0.01
    context_vec = torch.randn(vocab_size, dimension, dtype=torch.float32, device=device) * 0.01
    b = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    c = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    all_indices_i = torch.tensor([idx[0] for idx in non_zero_indices], dtype=torch.long, device=device)
    all_indices_j = torch.tensor([idx[1] for idx in non_zero_indices], dtype=torch.long, device=device)
    all_values = torch.tensor(non_zero_values, dtype=torch.float32, device=device)


    for epoch in range(epochs):
        total_loss = 0.0

        if device.type == "mps":  # GPU training
            for batch_start in range(0, len(non_zero_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(non_zero_indices))
                indices_i = all_indices_i[batch_start:batch_end]
                indices_j = all_indices_j[batch_start:batch_end]
                values = all_values[batch_start:batch_end]

                Xij = values
                wgt = torch.where(Xij < x_max, (Xij/x_max) ** alpha, torch.tensor(1.0, device=device))

                word_vecs_batch = word_vec[indices_i]
                context_vecs_batch = context_vec[indices_j]
                b_batch = b[indices_i]
                c_batch = c[indices_j]

                dots = torch.sum(word_vecs_batch * context_vecs_batch, dim=1)
                diff = dots + b_batch + c_batch - torch.log(Xij + 1)
                batch_loss = torch.sum(wgt * diff ** 2)
                total_loss += batch_loss.item()

                grad = 2 * wgt * diff
                grad_reshaped = grad.unsqueeze(1)
                word_vec.index_add_(0, indices_i, -learning_rate * grad_reshaped * context_vecs_batch)
                context_vec.index_add_(0, indices_j, -learning_rate * grad_reshaped * word_vecs_batch)
                b.index_add_(0, indices_i, -learning_rate * grad)
                c.index_add_(0, indices_j, -learning_rate * grad)

        else:  
            batches = []
            word_vec_np = word_vec.cpu().numpy()
            context_vec_np = context_vec.cpu().numpy()
            b_np = b.cpu().numpy()
            c_np = c.cpu().numpy()

            for batch_start in range(0, len(non_zero_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(non_zero_indices))
                args = (
                    all_indices_i[batch_start:batch_end].cpu().numpy(),
                    all_indices_j[batch_start:batch_end].cpu().numpy(),
                    all_values[batch_start:batch_end].cpu().numpy(),
                    word_vec_np, context_vec_np, b_np, c_np,
                    learning_rate, x_max, alpha
                )
                batches.append(args)

            with mp.Pool(processes=num_workers) as pool:
                losses = pool.map(cpu_batch, batches)
            total_loss = sum(losses)

           
            word_vec = torch.from_numpy(word_vec_np).to(device)
            context_vec = torch.from_numpy(context_vec_np).to(device)
            b = torch.from_numpy(b_np).to(device)
            c = torch.from_numpy(c_np).to(device)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {total_loss:.4f}")

    embeddings = (word_vec + context_vec).cpu().numpy()
    print("Training completed!")
    print(f"Final embeddings shape: {embeddings.shape}")
    return embeddings, word_to_id, id_to_word, metadata



def save_trained_model(embeddings, word_to_id, id_to_word, metadata,
                      training_config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/embeddings.npy", embeddings)
    with open(f"{output_dir}/word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)
    with open(f"{output_dir}/id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)
    complete_metadata = {**metadata, **training_config}
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(complete_metadata, f)
    print(f"Model saved to {output_dir}/")



if __name__ == "__main__":
    preprocessed_dir = "trymatrixes/tryenglish_matrix_30k"
    model_output_dir = "wordembeddings/glove_model_30k"
    dimension = 300
    epochs = 500
    learning_rate = 0.005
    batch_size = 10000

    print("Starting GloVe Training")
    print("=" * 40)

    embeddings, word_to_id, id_to_word, metadata = train_glove(
        preprocessed_dir,
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=6
    )

    if embeddings is not None:
        training_config = {
            'embedding_dim': dimension,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
        save_trained_model(embeddings, word_to_id, id_to_word, metadata,
                          training_config, model_output_dir)
        print(f"Complete! Model saved to: {model_output_dir}")
    else:
        print("Training failed.")
