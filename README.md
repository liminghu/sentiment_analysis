Sentiment analysis based on Transformer and Mamba architecture

Liming Hu

([dawninghu@gmail.com](mailto:dawninghu@gmail.com))

1. Data Source

The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is a standard benchmark containing 50,000 polarized movie reviews from IMDb designed for binary sentiment classification.

- **Total Labeled Reviews:** 50,000 reviews split evenly.
- **Training Set:** 25,000 movie reviews used to train models.
- **Testing Set:** 25,000 movie reviews used to test model performance.
- **Disjoint Split:** The training and testing sets contain completely different movies, ensuring a model cannot get high accuracy just by memorizing movie-specific names.
- **Binary Classes:** Labeled strictly as positive or negative.
- **Balanced Distribution:** Exactly 25,000 positive and 25,000 negative reviews across the sets.
- **Rating Thresholds:** Positive reviews have an IMDb score of ≥ 7/10, while negative reviews have a score of ≤ 4/10. Neutral reviews are left out of the labeled sets.
- **Unlabeled Documents:** Includes an extra 50,000 raw documents intended for unsupervised learning or pre-training.

For the 25K testing data, I split into 12.5k as validation, and the rest 12.5k as the untouched test data.

Here are representative examples of highly polarized reviews from the dataset that illustrate the distinct features of the positive and negative classes:

**Positive Review Example (Rating: ≥ 7/10)**

"This movie was an absolute masterpiece of suspense. The cinematography was breathtaking, using sharp contrasts and tight angles that made me feel trapped right along with the main characters. The lead actress delivered a powerful, nuanced performance that grounded the entire story. I was on the edge of my seat from the opening scene until the credits rolled. Highly recommended for anyone who appreciates smart storytelling."

**Negative Review Example (Rating: ≤ 4/10)**

"I want my two hours back. The plot was completely incoherent, jumping from one illogical scene to the next without any explanation. The dialogue felt incredibly forced and unnatural, making it impossible to connect with any of the flat, one-dimensional characters. Even the special effects looked cheap and outdated. Save your money and skip this total disaster."

**Structural Notes**

- **High Contrast:** The dataset deliberately excludes neutral reviews (5/10 and 6/10 ratings) to ensure strong, unambiguous language.
- **Text Length:** Reviews vary widely in length, ranging from single-sentence rants to multi-paragraph essays.

The absolute maximum length of a single raw movie review in the original IMDb dataset is **2,470 words** (which corresponds to **13,704 characters**). \[[1](https://github.com/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis/blob/main/Technical%20Report.md), [2](https://www.cliffsnotes.com/study-notes/27773244)\]

However, the dataset's text length varies heavily across individual reviews: \[[1](https://www.kaggle.com/code/bestealtiner/imdb-sentiment-analysis), [2](https://www.sciencedirect.com/org/science/article/pii/S1941629622000179)\]

- **Average length:** Approximately 230 to 255 words (~1,300 characters).
- **Minimum length:** Around 32 characters. \[[1](https://www.sciencedirect.com/org/science/article/pii/S1941629622000179), [2](https://www.cliffsnotes.com/study-notes/27773244)\]

**Practical Implementation Limits**

When people train machine learning models on this dataset, they rarely use the absolute maximum length because processing 2,470 words per sequence requires massive amounts of GPU memory. Instead, developers purposefully apply a **truncation limit** (max sequence length): \[[1](https://keras.io/api/datasets/imdb/), [2](https://www.kaggle.com/code/bestealtiner/imdb-sentiment-analysis), [3](https://medium.com/@ryblovartem/not-imdb-movie-reviews-dataset-eda-3042a0f3a38e), [4](https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/), [5](https://levelup.gitconnected.com/implementing-sentiment-classification-with-keras-507f582c80e4)\]

- **Standard Transformer Limits:** Models like BERT or DistilBERT have a hard architectural limit of **512 tokens**. Reviews longer than this are automatically cut off at the 512th token.

1. preprocessing

We directly load the data from hugging Face, and tokenize it using DistilBERT model:

print("Loading all 50,000 reviews from IMDb dataset...")

raw_datasets = load_dataset("stanfordnlp/imdb")

\# Split the original 25k test set 50/50 into distinct validation and test sets

split_test = raw_datasets\["test"\].train_test_split(test_size=0.5, seed=42)

raw_datasets\["validation"\] = split_test\["train"\] # 12,500 reviews

raw_datasets\["test"\] = split_test\["test"\] # 12,500 reviews

\# 3. Initialize the tokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):

&nbsp; return tokenizer(examples\["text"\], padding="max_length", truncation=True)

print("Tokenizing the entire dataset...")

\# Full allocation scale mapping profile across all dataset partitions

train_subset = raw_datasets\["train"\].map(tokenize_function, batched=True)

val_subset = raw_datasets\["validation"\].map(tokenize_function, batched=True)

test_subset = raw_datasets\["test"\].map(tokenize_function, batched=True)

\# Set up data formatting for PyTorch environments

train_subset.set_format(type="torch", columns=\["input_ids", "attention_mask", "label"\])

val_subset.set_format(type="torch", columns=\["input_ids", "attention_mask", "label"\])

test_subset.set_format(type="torch", columns=\["input_ids", "attention_mask", "label"\])

1. Performance Comparison

| Architecture                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Notes                          | Training                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Testing Accuracy |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| DistilBERT model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Pretrained, and then finetune. | learning_rate**\=**2e-5,<br><br>per_device_train_batch_size**\=**16,<br><br>per_device_eval_batch_size**\=**16,<br><br>num_train_epochs**\=**10,<br><br>weight_decay**\=**0.01,                                                                                                                                                                                                                                                                                                                                               | **0.9277**       |
| raw_custom_model **\=** CustomTransformerClassifier(<br><br>vocab_size**\=**tokenizer**.**vocab_size,<br><br>embed_dim**\=**128, _\# Lightweight dimension choice for faster custom layer computation_<br><br>num_heads**\=**4,<br><br>hidden_dim**\=**256,<br><br>num_layers**\=**2<br><br>)                                                                                                                                                                                                                                                                                      |                                | learning_rate**\=**1e-4, _\# Custom modules trained from scratch need higher learning rates than pre-trained ones_<br><br>per_device_train_batch_size**\=**16,<br><br>per_device_eval_batch_size**\=**16,<br><br>num_train_epochs**\=**10,<br><br>weight_decay**\=**0.01,<br><br>eval_strategy**\=**"epoch", _\# Compute validation metrics after every training epoch loop_<br><br>save_strategy**\=**"epoch",<br><br>load_best_model_at_end**\=True**, _\# Rollback to the checkpoint with the highest validation accuracy_ | 0.8348           |
| raw_custom_model **\=** TunedTransformerClassifier(<br><br>vocab_size**\=**tokenizer**.**vocab_size,<br><br>embed_dim**\=**256, _\# Scaled up embed_dim from 128 to 256 for higher capacity_<br><br>num_heads**\=**8, _\# Scaled up attention heads from 4 to 8_<br><br>hidden_dim**\=**512, _\# Scaled up feedforward network dimensions_<br><br>num_layers**\=**4, _\# Scaled up depth layers from 2 to 4_<br><br>dropout**\=**0.2 _\# Clear regularizer boundary_<br><br>)                                                                                                      |                                | learning_rate**\=**2e-4, _\# Optimized learning rate for scratch transformer convergence_<br><br>per_device_train_batch_size**\=**16,<br><br>per_device_eval_batch_size**\=**16,<br><br>num_train_epochs**\=**15, _\# Increased epochs from 5 to 8 for deeper weight optimization_<br><br>weight_decay**\=**0.02, _\# Increased weight decay to bound explosive parameter growth_<br><br>eval_strategy**\=**"epoch",<br><br>save_strategy**\=**"epoch",<br><br>load_best_model_at_end**\=True**,                              | **0.8673**       |
| Hybrid transformer and Mamba model:<br><br>Mamba:<br><br>self**.**mamba **\=** Mamba(<br><br>d_model**\=**embed_dim, _\# Model dimension_<br><br>d_state**\=**16, _\# SSM state expansion factor_<br><br>d_conv**\=**4, _\# Local convolution width_<br><br>expand**\=**2 _\# Block expansion factor_<br><br>)<br><br>raw_hybrid_model **\=** HybridMambaTransformerClassifier(<br><br>vocab_size**\=**tokenizer**.**vocab_size,<br><br>embed_dim**\=**256,<br><br>num_heads**\=**8,<br><br>hidden_dim**\=**512,<br><br>num_layers**\=**4,<br><br>dropout**\=**0.2 _#0.1_<br><br>) |                                | learning_rate**\=**5e-5, _\# Stable learning rate profile for scaling to large raw tokens_<br><br>per_device_train_batch_size**\=**32, _\# Increased batch size for processing high data density efficiently_<br><br>per_device_eval_batch_size**\=**32,<br><br>num_train_epochs**\=**15, _#8, # 3 full epochs across 25k samples provides deep convergence patterns_<br><br>weight_decay**\=**0.01,<br><br>eval_strategy**\=**"epoch",<br><br>save_strategy**\=**"epoch",<br><br>load_best_model_at_end**\=True**,           | 0.8414           |
| Pure Mamba Architecture:<br><br>raw_mamba_model **\=** PureMambaClassifier(<br><br>vocab_size**\=**tokenizer**.**vocab_size,<br><br>embed_dim**\=**256,<br><br>num_layers**\=**6, _\# Stacked layer depth_<br><br>dropout**\=**0.1<br><br>)                                                                                                                                                                                                                                                                                                                                        |                                | learning_rate**\=**5e-5,<br><br>per_device_train_batch_size**\=**32,<br><br>per_device_eval_batch_size**\=**32,<br><br>num_train_epochs**\=**15,<br><br>weight_decay**\=**0.01,<br><br>eval_strategy**\=**"epoch",<br><br>save_strategy**\=**"epoch",<br><br>load_best_model_at_end**\=True**,                                                                                                                                                                                                                                | 0.8785           |

1. Summary:

From the above table, we can see that the finetuned DistilBERT transformer architecture has the highest test accuracy, the pure Mamba architecture trained from scratch has quite a good performance, accuracy: 0.8785, and it can train and inference at high speed. The Customed transformer architecture trained from scratch has a good performance: 0.8673. The hybrid model of transformer and Mamba has not so bad performance: 0.84. I believe the performance can be improved if we have more data.

Basically, the scaling low still holds, the new Mamba architecture is picking up.

Notes: Google Gemini is used in draft the readme.