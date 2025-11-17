#!/usr/bin/env python3

def calculate_window_size(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print("File is empty.")
            return
        
        plot_lengths = []
        for line in lines:
            if "\t" in line:
                _, plot = line.split("\t", 1)
            else:
                plot = line
            words = plot.split()
            plot_lengths.append(len(words))
        
        avg_length = sum(plot_lengths) / len(plot_lengths)
        suggested_window = max(1, int(avg_length / 10))
        
        print(f"Number of plots: {len(lines)}")
        print(f"Average plot length (words): {avg_length:.2f}")
        print(f"Suggested window size for embeddings: {suggested_window}")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Set your file path here
    FILE_PATH = "english_final30k_train_mapped.txt"
    calculate_window_size(FILE_PATH)
