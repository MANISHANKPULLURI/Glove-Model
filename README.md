# **GloVe Model — NumPy Implementation**

A from-scratch implementation of the GloVe (Global Vectors) algorithm using only NumPy, trained separately on Hindi and English corpora.

---

## **Overview**

This project implements a fully custom GloVe embedding pipeline using only NumPy.
Hindi and English corpora were processed independently to build two separate embedding models.
The workflow includes corpus preprocessing, co-occurrence matrix construction, GloVe optimization, embedding extraction, and downstream evaluation using a CNN classifier.

---

## **Key Features**

* Pure NumPy-based GloVe implementation (no PyTorch/TensorFlow)
* Separate Hindi and English embedding models
* Training corpus sizes: 2.5k, 15k, 30k
* Configurable:

  * Window size
  * Embedding size (50, 100, 200) for Quality
  * Learning rate
* Full co-occurrence matrix construction
* Intrinsic evaluation using cosine similarity
* CNN classifier for downstream NLP evaluation

---

## **Dataset & Model Files**

Complete datasets, co-occurrence matrices, embeddings, and trained GloVe model `.pkl` files for **all corpus sizes** are available here:

📁 **Datasets + Co-Occurrence Matrices + Embeddings + Trained Models**
[https://drive.google.com/drive/folders/1lsvSuEEIXpA1bPFba7jkiMSiSSwRPdTz?usp=drive_link](https://drive.google.com/drive/folders/1lsvSuEEIXpA1bPFba7jkiMSiSSwRPdTz?usp=drive_link)

All files can be directly used for testing by setting your desired paths inside the code.

Also check the **Results** folder in this GitHub repository to view intrinsic and extrinsic evaluation outputs.

---

## **GloVe Implementation Steps**

### **1. Vocabulary & Token Processing**

Hindi and English corpora were processed independently:

* Lowercasing
* Tokenization
* Punctuation removal
* Vocabulary indexing
* OOV handling per corpus size

---

### **2. Co-Occurrence Matrix Construction**

The first major step was constructing the **co-occurrence matrix**:

* Sliding context window (eg :sizes 2, 4, 5) for context 
* Weighted by inverse distance
* Stored efficiently (sparse form)
* Forms the statistical foundation of GloVe training

---

### **3. Training Word Embeddings**

Embeddings were generated using the classic GloVe optimization objective:

* used Co-occurence Matrix to Build Embeddings
* Embedding dimensions tested: 50, 100, 200
* Manual gradient descent implementation

---

### **4. Model Training**

Fed Embeddings,TrainDataset To CNN and Evaluated on the Test Datset

---


---

## **Evaluation Approach**

### **1. Intrinsic Evaluation**

Cosine similarity was computed for:

* Hindi → Hindi word relationships
* English → English word relationships

---

### **2. Extrinsic Evaluation**

A CNN classifier was trained using the generated embeddings to evaluate downstream usefulness.

Metrics measured:

* Accuracy
* Precision
* Recall

---

## **TechStack Used**

* Python
* Cupy - for cuda(glovetrainingcuda.py)
* metal - for macos(glovetrainingmetal.py)
* NumPy
* pandas
* Matplotlib
* Custom CNN classifier

---

## **Future Work**

* Add subsampling for high-frequency tokens
* Train on larger Hindi/English corpora

---


