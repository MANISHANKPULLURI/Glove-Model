import numpy as np
import pickle
import os

try:
    import cupy as cp
    gpu_available = True
    print(" CuPy available, using GPU (CUDA/MPS if configured).")
except ImportError:
    cp = np  
    gpu_available = False
    print(" CuPy not available, using NumPy (CPU only).")


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


def safe_gpu_init():
    if not gpu_available:
        print("CuPy not available → running on CPU.")
        return False
    try:
        print("Initializing GPU...")
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Found {device_count} GPU(s)")
        cp.cuda.Device(0).use()
        test = cp.array([1, 2, 3])
        result = cp.sum(test)
        print(f"GPU available: {cp.cuda.Device()}")
        print(f"Test operation successful: {result}")
        del test
        try:
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU memory cleared")
        except Exception as mem_error:
            print(f"Could not clear memory pool: {mem_error}")
            print("Continuing anyway...")
        return True
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        return False


def train_glove_gpu(data_dir, dimension=100, epochs=100, learning_rate=0.01,
                   x_max=100, alpha=0.75, batch_size=30000):
    co_occurrence_data, word_to_id, id_to_word, metadata = load_preprocessed_data(data_dir)

    if not safe_gpu_init():
        print("Proceeding with CPU fallback (no GPU acceleration).")

    non_zero_indices = co_occurrence_data['indices']
    non_zero_values = co_occurrence_data['values']
    vocab_size = metadata['vocab_size']

    print(f"Training Configuration:")
    print(f"  - Embedding dimension: {dimension}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")

    def weighting_function(x):
        return cp.where(x < x_max, (x/x_max) ** alpha, 1.0)

    print(f"Initializing embeddings ({vocab_size} x {dimension})...")
    try:
        cp.random.seed(0)
        word_vec = cp.random.uniform(-0.01, 0.01, (vocab_size, dimension)).astype(cp.float32)
        context_vec = cp.random.uniform(-0.01, 0.01, (vocab_size, dimension)).astype(cp.float32)
        b = cp.zeros(vocab_size, dtype=cp.float32)
        c = cp.zeros(vocab_size, dtype=cp.float32)
        print("Embeddings initialized on GPU/CPU")
    except Exception as e:
        print(f"Failed to allocate embeddings: {e}")
        print("Try reducing dimension size")
        return None, None, None, None

    print("Converting co-occurrence data to GPU/CPU...")
    try:
        all_indices_i = cp.array([idx[0] for idx in non_zero_indices], dtype=cp.int32)
        all_indices_j = cp.array([idx[1] for idx in non_zero_indices], dtype=cp.int32)
        all_values = cp.array(non_zero_values, dtype=cp.float32)
        print(f"{len(non_zero_indices)} co-occurrences loaded")
    except Exception as e:
        print(f"Failed to load co-occurrence data: {e}")
        return None, None, None, None

    def print_gpu_memory():
        if gpu_available:
            mempool = cp.get_default_memory_pool()
            print(f"GPU memory used: {mempool.used_bytes() / 1024**3:.2f} GB")

    print_gpu_memory()
    print(f"Starting training...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_start in range(0, len(non_zero_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(non_zero_indices))
            indices_i = all_indices_i[batch_start:batch_end]
            indices_j = all_indices_j[batch_start:batch_end]
            values = all_values[batch_start:batch_end]

            Xij = values
            wgt = weighting_function(Xij)

            word_vecs_batch = word_vec[indices_i]
            context_vecs_batch = context_vec[indices_j]
            b_batch = b[indices_i]
            c_batch = c[indices_j]

            dots = cp.sum(word_vecs_batch * context_vecs_batch, axis=1)
            diff = dots + b_batch + c_batch - cp.log(Xij + 1)
            batch_loss = cp.sum(wgt * diff ** 2)
            total_loss += float(batch_loss)

            grad = 2 * wgt * diff
            grad_reshaped = grad.reshape(-1, 1)
            word_vec_updates = grad_reshaped * context_vecs_batch
            context_vec_updates = grad_reshaped * word_vecs_batch

            cp.add.at(word_vec, indices_i, -learning_rate * word_vec_updates)
            cp.add.at(context_vec, indices_j, -learning_rate * context_vec_updates)
            cp.add.at(b, indices_i, -learning_rate * grad)
            cp.add.at(c, indices_j, -learning_rate * grad)

            del word_vecs_batch, context_vecs_batch, b_batch, c_batch
            del dots, diff, grad, grad_reshaped, word_vec_updates, context_vec_updates

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {total_loss:.4f}")
            print_gpu_memory()

    embeddings = word_vec + context_vec
    embeddings_cpu = cp.asnumpy(embeddings) if gpu_available else embeddings

    print(f"Training completed!")
    print(f"Final embeddings shape: {embeddings_cpu.shape}")
    return embeddings_cpu, word_to_id, id_to_word, metadata


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
    preprocessed_dir = "matrixes/hindi_matrix_30k"
    model_output_dir = "embeddings/glove_model_hindi_30k"
    dimension = 150
    epochs = 300
    learning_rate = 0.005
    batch_size = 10000

    print("Starting GloVe Training")
    print("=" * 40)

    result = train_glove_gpu(
        preprocessed_dir,
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    embeddings, word_to_id, id_to_word, metadata = result
    if embeddings is not None:
        print("Training successful! Saving model...")
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
        print("Training failed!")
        print("Next steps:")
        print("1. Fix GPU issues (restart system)")
        print("2. Or reduce dimension/batch_size")
        print("3. Or try CPU-based training instead")
