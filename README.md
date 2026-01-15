
# 🚀 Generative AI Systems

This repository presents a **comprehensive, end-to-end exploration of Generative AI systems**, progressing from **classical NLP models** to **large language models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, **Reinforcement Learning from Human Feedback (RLHF)**, and **Generative Adversarial Networks (GANs)**.

Rather than treating each task in isolation, this project demonstrates how modern GenAI systems are **built, trained, evaluated, and aligned**, using **real datasets, modern architectures, and industry-relevant techniques**.

---

## 🛠 Technologies Used

- PyTorch
- Hugging Face Transformers & Datasets
- PEFT (LoRA)
- TRL (PPO)
- FAISS
- Sentence-Transformers
- BERTopic
- MedMNIST
- CUDA / GPU acceleration

---

## 🧠 Motivation

Modern Generative AI is not just about prompting LLMs.  
Real-world systems require:

- Careful **data preprocessing**
- **Model fine-tuning** and **parameter efficiency**
- **Prompt engineering and reasoning control**
- **Retrieval over large knowledge bases**
- **Human preference alignment (RLHF)**
- **Robust evaluation (automatic + human)**
- **Generative modeling beyond text (images)**

This repository brings **all of these components together** in one unified pipeline.

---

## 📌 Structure

```
├── HW1_NLP_Foundations/
│   ├── Transformers, RNNs, embeddings, zero-shot LLMs
│
├── HW2_Prompting_and_Representation/
│   ├── Prompt engineering, Siamese networks, topic modeling
│
├── HW3_LLM_Finetuning_and_RAG/
│   ├── LLM fine-tuning (LoRA), FAISS-based RAG
│
├── HW4_RLHF_and_GANs/
│   ├── Reward modeling, PPO, LoRA PPO, DCGAN
│
└── README.md
```

---

## 🔹 Part 1: NLP & Transformer Foundations

### What was done
- RNN / LSTM / GRU models with **Word2Vec, GloVe, FastText**
- Transformer models: **DistilBERT, DistilRoBERTa**
- Zero-shot LLMs: **GPT-4o-mini, LLaMA**
- Error analysis and tokenization comparison

### Key Insight
Fine-tuned transformers outperform classical RNNs, while zero-shot LLMs generalize well with no training but higher inference cost.

---

## 🔹 Part 2: Prompt Engineering & Representation Learning

### Prompt Engineering
- Models: **Mistral-7B, LLaMA-3.1**
- Techniques: Zero-shot, Few-shot, CoT, Self-consistency, Tree-of-Thought

**Insight:** Explicit reasoning can degrade semantic similarity tasks.

### Siamese Network
- Sentence embeddings + cosine similarity
- Contrastive loss training

### Topic Modeling
- BERTopic + SBERT
- LLM-as-a-judge for semantic relevance

---

## 🔹 Part 3: LLM Fine-Tuning & RAG

### LLM Fine-Tuning
- Tabular → text prompts (Titanic dataset)
- Models: **Mistral-7B (LoRA)**, **DeepSeek-R1-1.5B**
- Smaller model generalized better

### Retrieval-Augmented Generation
- arXiv ML papers (117k+)
- Sentence chunking, FAISS indexing
- Naive RAG vs Re-ranking RAG

---

## 🔹 Part 4: RLHF + PPO + GANs

### Reward Model
- Dataset: Stanford Human Preferences
- Model: Qwen2.5-0.5B + scalar reward head

### PPO Fine-Tuning
- Full PPO vs LoRA PPO
- LoRA PPO showed better stability and alignment

### DCGAN (Bonus)
- Dataset: ChestMNIST
- 1000 epochs training
- Generated 32 realistic chest X-ray images

---

## 🎯 Key Takeaways

- Smaller models can outperform larger ones with better training
- Prompting is powerful but not always optimal
- Retrieval is essential for factual generation
- LoRA > full fine-tuning for RLHF
- GANs remain useful beyond text

---

## 👩‍💻 Author

**Mrunali Katta**  
Master’s Student – Data Analytics  
Focus: Generative AI, LLMs, RLHF, RAG, Deep Learning
