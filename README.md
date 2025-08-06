# 🧠 Smart Refrigerator AI: Food Inventory & Recipe Recommender  
### IE7374, Summer 2025 – Group 12

## 📚 Table of Contents
- [About the Project](#about-the-project)
- [Architecture Overview](#architecture-overview)
- [Dataset](#dataset)
- [Setup & Running the App](#setup--running-the-app)
- [Research Questions](#research-questions)
- [Contributors](#contributors)
- [References](#references)

---

## 📌 About the Project

In response to the global issue of food waste — over 1 billion meals discarded daily (UNEP, 2024) — this project implements a **smart kitchen assistant** that uses **generative AI** and **retrieval-augmented generation (RAG)** to help users:

- Track refrigerator inventory
- Get alerts for expiring items
- Generate personalized recipes using available ingredients

Rather than using expensive APIs or black-box services, this project uses **open-source GPT-2**, fine-tuned on the **RecipeNLG dataset**, to deliver an end-to-end application that emphasizes **sustainability**, **user-friendliness**, and **practical AI integration**.

---

## 🧠 GPT-2 Architecture & Integration – Smart Refrigerator AI

This document provides a technical overview of the GPT-2 model architecture, how it was adapted for recipe generation, and how it integrates with other layers (LangChain, Streamlit, databases) in the Smart Refrigerator AI system.

---

### 🧩 1. GPT-2 Core Architecture

GPT-2 is a **decoder-only Transformer model** designed for generative tasks like language modeling.
Please find our research article for reference located [here](documentation/GPT2_Architecture_Research.docx)

#### Key Components

- **Token Embeddings & Positional Encodings**  
  Convert input text into token vectors and inject positional info for word order.

- **Multi-Head Self-Attention**  
  Learns dependencies between words in a sequence by attending to all tokens at once.

- **Feed-Forward Networks (FFN)**  
  Non-linear layers applied after attention for deeper representation.

- **Residual Connections & Layer Normalization**  
  Prevent vanishing gradients and stabilize training.

- **Output Softmax Layer**  
  Predicts the probability of the next token in a sequence.

#### Training Objective

- **Causal Language Modeling**  
  Predicts the next token using only prior context (left-to-right).  
  Trained by minimizing cross-entropy loss on large text corpora.

- **Fine-Tuning**  
  For this project, GPT-2 is fine-tuned using the [RecipeNLG dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) to map ingredient inputs to coherent recipe outputs.

### 🔁 Integration in the Application Stack

GPT-2 is the core **recipe generation engine**, surrounded by supporting modules for data flow, retrieval, and user interaction.

#### End-to-End Flow

1. **User Input** via Streamlit – User enters available ingredients.
2. **Inventory DB** (DuckDB or SQLite) – Tracks items and expiration.
3. **GPT-2 Model** – Generates new recipes based on user input.
4. **User Feedback** – Accept/reject logic used for possible future model refinement.

---

## 🔁 2. Integration in the Application Stack

GPT-2 is the core **recipe generation engine**, surrounded by supporting modules for data flow, retrieval, and user interaction.

### End-to-End Flow

1. **User Input** via Streamlit – User enters available ingredients.
2. **Inventory DB** (DuckDB or SQLite) – Tracks items and expiration.
3. **GPT-2 Model** – Generates new recipes based on user input.
4. **User Feedback** – Accept/reject logic used for possible future model refinement.

---

## 🧱 3. Architecture Diagram



---
## 🏗️ Architecture Overview

Our system is a pipeline integrating:
- **Manual inventory input** (via Streamlit interface)
- **GPT-2 model fine-tuned** with RecipeNLG for generating recipes
- **Lightweight storage** (DuckDB or SQLite) to track inventory and updates

> 📌 Future expansion: Optional integration of vision-based models for scanning fridge contents.

---

## 📊 Dataset

We use the [**RecipeNLG dataset**](https://www.kaggle.com/datasets/paultimothymooney/recipenlg), a high-quality corpus of over 2 million recipes including ingredients, instructions, and named entities. Compared to prior datasets like Recipe1M+, RecipeNLG features:

- Cleaned fractions and measurements
- Accurate control-token structure (e.g., `<INGR_START>`, `<TITLE_END>`)
- Optimized for **semi-structured generation tasks**

### 🔢 Recipe Format
```json
{
  "title": "Spicy Stuffed Peppers",
  "ingredients": ["3/4 lbs. lean beef", "1 cup rice", ...],
  "directions": ["Combine ingredients.", "Bake for 30 minutes."],
  "source": "www.food.com",
  "ner": ["beef", "rice"]
}
```

## 🤖 Modeling & UI Interface

### GPT-2 Fine-Tuning

- **Input Format**: Ingredient list with control tokens  
  Example: `<INPUT_START> beef <NEXT_INPUT> onion <INPUT_END>`  
- **Output**: Title, Ingredients, Instructions  
- **Training**: Hugging Face Transformers using Colab  
- **Cleaning**: Removed short or vague recipes; ensured consistency

### Streamlit App Interface

- Users input fridge contents  
- Get recipe recommendations (accept/reject)  
- Notifications for expiring items  
- Interact with model via chat-like interface  

---

## ⚙️ Setup & Running the App

### 🔧 Requirements

To get started, install the necessary dependencies:
[The Requirements File is located at this link.](https://github.com/jasonhops/IE7374/blob/main/code/streamlit/requirements.txt)
```bash
pip install -r requirements.txt
```

### Model Setup

Modify the home.py file with the path of your trained model paths on line 7 of the file.
```python
    model_path = "./gpt2-recipes-manual_at_80pct" #Change this to your location of your trained model
```
### ▶️ Launch the Interface

Run the Streamlit app using the following command:

```bash
streamlit run home.py
```

---

### 📂 Future File Overview

| File / Folder       | Description                                      |
|---------------------|--------------------------------------------------|
| `home.py`  | Streamlit interface for interacting with the AI  |
| `gpt2_trainer.py`   | Script for fine-tuning the GPT-2 model           |
| `dataset/`          | Folder containing cleaned RecipeNLG subset       |
| `code/`             | Saved fine-tuned GPT-2 weights                   |
| `documentation/`    | Key project details and articles                 |

---

## 👥 Contributors

- **Ahantya Vempati** – Architecture Research & Proofreading  
- **Chantelle Chan** – Data Preprocessing & Testing  
- **Jensen Ho** – GPT-2 Training & Hyperparameter Tuning  
- **Shyam Patel** – Streamlit App & Front End Development

---

## 📌 References

- United Nations Environment Programme. (2024). *World squanders over 1 billion meals a day*.  
  [https://www.unep.org/news-and-stories/press-release/world-squanders-over-1-billion-meals-day-un-report](https://www.unep.org/news-and-stories/press-release/world-squanders-over-1-billion-meals-day-un-report)

- Bien et al. (2020). *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation*.  
  [https://recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl)

- Kaggle Dataset:  
  [https://www.kaggle.com/datasets/paultimothymooney/recipenlg](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)

