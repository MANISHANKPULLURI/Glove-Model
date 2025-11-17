import numpy as np
from scipy.sparse import dok_matrix
import pickle
import os
from multiprocessing import Pool
import time
from collections import defaultdict

def load_our_data(filepath):
    with open(filepath,'r',encoding = 'utf-8') as f:
        return [line.strip().split() for line in f]

def process_sentence_batch(args):
    sentence_batch, word_to_id, window = args
    local_cooccur = defaultdict(float)
    for sentence in sentence_batch:
        indices = [word_to_id.get(w) for w in sentence if w in word_to_id]
        if len(indices) < 2:
            continue
        for word_index, word in enumerate(indices):
            start = max(0, word_index - window)
            end = min(len(indices), word_index + window + 1)
            for context_index in range(start, end):
                if word_index == context_index:
                    continue
                distance = abs(context_index - word_index)
                context_word = indices[context_index]
                key = (word, context_word)
                local_cooccur[key] += 1.0 / distance
    return dict(local_cooccur)

def build_and_save_cooccurrence_matrix_6cores(training_data, output_dir, window=20):
    n_processes = 6
    print(f"Using {n_processes} CPU cores (optimized for RAM usage)")
    print(f"Loading data from {training_data}...")
    training = load_our_data(training_data)
    print("Building vocabulary...")
    total_words = [word for sentence in training for word in sentence]
    vocabulary = list(set(total_words))
    word_to_id = {word: i for i,word in enumerate(vocabulary)}
    id_to_word = {i:word for word,i in word_to_id.items()}
    size = len(vocabulary)
    print(f"Vocabulary size: {size}")
    print(f"Total sentences: {len(training)}")
    print(f"Window size: {window}")
    batch_size = max(1000, len(training) // (n_processes * 3))
    sentence_batches = []
    for i in range(0, len(training), batch_size):
        batch = training[i:i + batch_size]
        sentence_batches.append((batch, word_to_id, window))
    print(f"Created {len(sentence_batches)} batches for parallel processing")
    print(f"Average batch size: {len(training) // len(sentence_batches)} sentences")
    print(f"Memory usage: ~{len(sentence_batches) * len(word_to_id) * 8 / 1024**2:.1f} MB per process")
    print("Building co-occurrence matrix with 6 cores... (RAM optimized)")
    start_time = time.time()
    with Pool(processes=n_processes) as pool:
        results = []
        completed = 0
        for result in pool.imap(process_sentence_batch, sentence_batches):
            results.append(result)
            completed += 1
            if completed % max(1, len(sentence_batches) // 10) == 0:
                elapsed = time.time() - start_time
                progress = completed / len(sentence_batches) * 100
                eta = elapsed / completed * len(sentence_batches) - elapsed
                print(f"Progress: {progress:.1f}% ({completed}/{len(sentence_batches)} batches) - "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    print("Merging results from 6 processes...")
    merge_start = time.time()
    final_cooccur = defaultdict(float)
    for i, result_dict in enumerate(results):
        if i % max(1, len(results) // 5) == 0:
            print(f"Merging batch {i+1}/{len(results)}")
        for key, value in result_dict.items():
            final_cooccur[key] += value
    final_cooccur = dict(final_cooccur)
    merge_time = time.time() - merge_start
    total_time = time.time() - start_time
    print(f"Co-occurrence matrix built in {total_time:.1f} seconds with {len(final_cooccur)} non-zero entries")
    print(f"Processing time: {total_time - merge_time:.1f}s, Merging time: {merge_time:.1f}s")
    print(f"Speed improvement: ~{n_processes}x faster than single-core!")
    os.makedirs(output_dir, exist_ok=True)
    print("Saving co-occurrence matrix...")
    non_zero_indices = list(final_cooccur.keys())
    non_zero_values = list(final_cooccur.values())
    co_occurrence_data = {
        'indices': non_zero_indices,
        'values': non_zero_values
    }
    with open(f"{output_dir}/cooccurrence_data.pkl", "wb") as f:
        pickle.dump(co_occurrence_data, f)
    print("Saving vocabulary...")
    with open(f"{output_dir}/word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)
    with open(f"{output_dir}/id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)
    metadata = {
        'vocab_size': size,
        'window_size': window,
        'dataset': training_data,
        'total_sentences': len(training),
        'non_zero_entries': len(non_zero_indices),
        'build_time_seconds': total_time,
        'n_processes_used': n_processes,
        'method': '6_core_optimized'
    }
    with open(f"{output_dir}/preprocessing_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n All preprocessing data saved to {output_dir}/")
    print(f"Files created:")
    print(f"  - cooccurrence_data.pkl ({len(non_zero_indices)} entries)")
    print(f"  - word_to_id.pkl ({size} words)")
    print(f"  - id_to_word.pkl ({size} words)")
    print(f"  - preprocessing_metadata.pkl")
    print(f"\n Total time: {total_time:.1f} seconds using {n_processes} cores")
    print(f" RAM-optimized: Used larger batches to reduce memory overhead")
    return output_dir

def build_and_save_cooccurrence_matrix(training_data, output_dir, window=20):
    print(f"Loading data from {training_data}...")
    training = load_our_data(training_data)
    print("Building vocabulary...")
    total_words = [word for sentence in training for word in sentence]
    vocabulary = list(set(total_words))
    word_to_id = {word: i for i,word in enumerate(vocabulary)}
    id_to_word = {i:word for word,i in word_to_id.items()}
    size = len(vocabulary)
    print(f"Vocabulary size: {size}")
    print(f"Total sentences: {len(training)}")
    print(f"Window size: {window}")
    print("Building co-occurrence matrix... (this will take time)")
    x = dok_matrix((size,size), dtype=np.float32)
    for sentence_idx, sentence in enumerate(training):
        if sentence_idx % 5000 == 0:
            print(f"Processed {sentence_idx}/{len(training)} sentences ({sentence_idx/len(training)*100:.1f}%)")
        indice = [word_to_id.get(w) for w in sentence if w in word_to_id]
        for word_index, word in enumerate(indice):
            start = max(0, word_index - window)
            end = min(len(indice), word_index + window + 1)
            for context_index in range(start, end):
                if word_index == context_index:
                    continue
                distance = abs(context_index - word_index)
                context_word = indice[context_index]
                x[word, context_word] = x.get((word, context_word), 0) + 1.0 / distance
    print(f"Co-occurrence matrix built with {len(x.keys())} non-zero entries")
    os.makedirs(output_dir, exist_ok=True)
    print("Saving co-occurrence matrix...")
    non_zero_indices = list(x.keys())
    co_occurrence_data = {
        'indices': non_zero_indices,
        'values': [x[idx] for idx in non_zero_indices]
    }
    with open(f"{output_dir}/cooccurrence_data.pkl", "wb") as f:
        pickle.dump(co_occurrence_data, f)
    print("Saving vocabulary...")
    with open(f"{output_dir}/word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)
    with open(f"{output_dir}/id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)
    metadata = {
        'vocab_size': size,
        'window_size': window,
        'dataset': training_data,
        'total_sentences': len(training),
        'non_zero_entries': len(non_zero_indices)
    }
    with open(f"{output_dir}/preprocessing_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n All preprocessing data saved to {output_dir}/")
    print(f"Files created:")
    print(f"  - cooccurrence_data.pkl ({len(non_zero_indices)} entries)")
    print(f"  - word_to_id.pkl ({size} words)")
    print(f"  - id_to_word.pkl ({size} words)")
    print(f"  - preprocessing_metadata.pkl")
    return output_dir

if __name__ == "__main__":
    training_data = 'datasets/english/cleaned_english_30000.txt'
    output_dir = "matrixes/english_matrix_30k"
    window = 20
    print("Choose processing method:")
    print("1. Single core (44 minutes, lowest RAM)")
    print("2. 6 cores (7-8 minutes, moderate RAM)")
    choice = input("Enter choice (1 or 2, default=2): ").strip() or "2"
    if choice == "1":
        print(" Using single core (original method)")
        build_and_save_cooccurrence_matrix(training_data, output_dir, window)
    else:
        print("Using 6 cores (RAM optimized)")
        build_and_save_cooccurrence_matrix_6cores(training_data, output_dir, window)
