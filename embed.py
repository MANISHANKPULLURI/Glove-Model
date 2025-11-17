import numpy as np

file_path = "models/text_to_word_embeddings/glove_model_2.5k/embeddings.npy"
data = np.load(file_path)

print("Shape of array:", data.shape)
print("First 5 rows:\n", data[:1])
