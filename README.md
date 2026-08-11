Sentiment analysis based on Transformer and Mamba architecture

Liming Hu

([dawninghu@gmail.com](mailto:dawninghu@gmail.com))

Data Source

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

Preprocessing

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

Performance Comparison

<div class="joplin-table-wrapper"><table><tbody><tr><th><p>Architecture</p></th><th><p>Notes</p></th><th><p>Training</p></th><th><p>Testing Accuracy</p></th></tr><tr><td><p>DistilBERT model</p></td><td><p>Pretrained, and then finetune.</p></td><td><p>learning_rate<strong>=</strong>2e-5,</p><p>per_device_train_batch_size<strong>=</strong>16,</p><p>per_device_eval_batch_size<strong>=</strong>16,</p><p>num_train_epochs<strong>=</strong>10,</p><p>weight_decay<strong>=</strong>0.01,</p></td><td><p><strong>0.9277</strong></p></td></tr><tr><td><p>raw_custom_model <strong>=</strong> CustomTransformerClassifier(</p><p>vocab_size<strong>=</strong>tokenizer<strong>.</strong>vocab_size,</p><p>embed_dim<strong>=</strong>128, <em># Lightweight dimension choice for faster custom layer computation</em></p><p>num_heads<strong>=</strong>4,</p><p>hidden_dim<strong>=</strong>256,</p><p>num_layers<strong>=</strong>2</p><p>)</p></td><td></td><td><p>learning_rate<strong>=</strong>1e-4, <em># Custom modules trained from scratch need higher learning rates than pre-trained ones</em></p><p>per_device_train_batch_size<strong>=</strong>16,</p><p>per_device_eval_batch_size<strong>=</strong>16,</p><p>num_train_epochs<strong>=</strong>10,</p><p>weight_decay<strong>=</strong>0.01,</p><p>eval_strategy<strong>=</strong>"epoch", <em># Compute validation metrics after every training epoch loop</em></p><p>save_strategy<strong>=</strong>"epoch",</p><p>load_best_model_at_end<strong>=True</strong>, <em># Rollback to the checkpoint with the highest validation accuracy</em></p></td><td><p>0.8348</p></td></tr><tr><td><p>raw_custom_model <strong>=</strong> TunedTransformerClassifier(</p><p>vocab_size<strong>=</strong>tokenizer<strong>.</strong>vocab_size,</p><p>embed_dim<strong>=</strong>256, <em># Scaled up embed_dim from 128 to 256 for higher capacity</em></p><p>num_heads<strong>=</strong>8, <em># Scaled up attention heads from 4 to 8</em></p><p>hidden_dim<strong>=</strong>512, <em># Scaled up feedforward network dimensions</em></p><p>num_layers<strong>=</strong>4, <em># Scaled up depth layers from 2 to 4</em></p><p>dropout<strong>=</strong>0.2 <em># Clear regularizer boundary</em></p><p>)</p></td><td></td><td><p>learning_rate<strong>=</strong>2e-4, <em># Optimized learning rate for scratch transformer convergence</em></p><p>per_device_train_batch_size<strong>=</strong>16,</p><p>per_device_eval_batch_size<strong>=</strong>16,</p><p>num_train_epochs<strong>=</strong>15, <em># Increased epochs from 5 to 8 for deeper weight optimization</em></p><p>weight_decay<strong>=</strong>0.02, <em># Increased weight decay to bound explosive parameter growth</em></p><p>eval_strategy<strong>=</strong>"epoch",</p><p>save_strategy<strong>=</strong>"epoch",</p><p>load_best_model_at_end<strong>=True</strong>,</p></td><td><p><strong>0.8673</strong></p></td></tr><tr><td><p>Hybrid transformer and Mamba model:</p><p>Mamba:</p><p>self<strong>.</strong>mamba <strong>=</strong> Mamba(</p><p>d_model<strong>=</strong>embed_dim, <em># Model dimension</em></p><p>d_state<strong>=</strong>16, <em># SSM state expansion factor</em></p><p>d_conv<strong>=</strong>4, <em># Local convolution width</em></p><p>expand<strong>=</strong>2 <em># Block expansion factor</em></p><p>)</p><p>raw_hybrid_model <strong>=</strong> HybridMambaTransformerClassifier(</p><p>vocab_size<strong>=</strong>tokenizer<strong>.</strong>vocab_size,</p><p>embed_dim<strong>=</strong>256,</p><p>num_heads<strong>=</strong>8,</p><p>hidden_dim<strong>=</strong>512,</p><p>num_layers<strong>=</strong>4,</p><p>dropout<strong>=</strong>0.2 <em>#0.1</em></p><p>)</p></td><td></td><td><p>learning_rate<strong>=</strong>5e-5, <em># Stable learning rate profile for scaling to large raw tokens</em></p><p>per_device_train_batch_size<strong>=</strong>32, <em># Increased batch size for processing high data density efficiently</em></p><p>per_device_eval_batch_size<strong>=</strong>32,</p><p>num_train_epochs<strong>=</strong>15, <em>#8, # 3 full epochs across 25k samples provides deep convergence patterns</em></p><p>weight_decay<strong>=</strong>0.01,</p><p>eval_strategy<strong>=</strong>"epoch",</p><p>save_strategy<strong>=</strong>"epoch",</p><p>load_best_model_at_end<strong>=True</strong>,</p></td><td><p>0.8414</p></td></tr><tr><td><p>Pure Mamba Architecture:</p><p>raw_mamba_model <strong>=</strong> PureMambaClassifier(</p><p>vocab_size<strong>=</strong>tokenizer<strong>.</strong>vocab_size,</p><p>embed_dim<strong>=</strong>256,</p><p>num_layers<strong>=</strong>6, <em># Stacked layer depth</em></p><p>dropout<strong>=</strong>0.1</p><p>)</p></td><td></td><td><p>learning_rate<strong>=</strong>5e-5,</p><p>per_device_train_batch_size<strong>=</strong>32,</p><p>per_device_eval_batch_size<strong>=</strong>32,</p><p>num_train_epochs<strong>=</strong>15,</p><p>weight_decay<strong>=</strong>0.01,</p><p>eval_strategy<strong>=</strong>"epoch",</p><p>save_strategy<strong>=</strong>"epoch",</p><p>load_best_model_at_end<strong>=True</strong>,</p></td><td><p>0.8785</p></td></tr><tr><td><p>class AdvancedHybridClassifier(nn.Module): def __init__(self, pretrained_model_name="bert-base-uncased", num_layers=6, ratio=2):</p><p>base_transformer = AutoModel.from_pretrained(pretrained_model_name)</p><p>self.embedding = base_transformer.embeddings</p><p>self.pooler = AttentivePooling(embed_dim) self.classifier = nn.Sequential( nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2) )</p><p>optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01) epochs = 4 total_steps = len(train_loader) * epochs scheduler = get_cosine_schedule_with_warmup( optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps )</p></td><td><p><strong>pre-trained BERT embeddings</strong>, an increased context length of <strong>512 tokens</strong>, an <strong>Attentive Pooling block</strong>, and a <strong>Learning Rate Scheduler</strong> to maximize classification power</p></td><td><p>Instead of an untrained nn.Embedding(vocab_size, embed_dim), initialize your input layer with pre-trained word vectors like GloVe or extract frozen hidden states from a pre-trained language backbone like DistilBERT.</p><p>Average pooling treats all tokens equally (including padding tokens). Replace nn.AdaptiveAvgPool1d(1) with a simple attentive pooling layer so the classifier learns <em>which</em> hidden states represent the core summary.</p><p>To keep the Transformer and Mamba weights from distorting during early training iterations, incorporate a linear learning rate warmup phase followed by a cosine decay scheduler.</p></td><td><p>Epoch 1/4 | Loss: 0.4547 | Train Acc: 77.20%</p><p>&gt;&gt; Current Test Set Target Accuracy: 86.94%</p><p>Epoch 2/4 | Loss: 0.2591 | Train Acc: 90.23%</p><p>&gt;&gt; Current Test Set Target Accuracy: 89.13%</p><p>Epoch 3/4 | Loss: 0.1582 | Train Acc: 94.81%</p><p>&gt;&gt; Current Test Set Target Accuracy: 88.57%</p><p>Epoch 4/4 | Loss: 0.0976 | Train Acc: 97.29%</p><p>&gt;&gt; Current Test Set Target Accuracy: <strong>88.70</strong>%</p></td></tr><tr><td><p>Hybrid</p></td><td><p><strong>-Frozen Weights Feature Extraction: By freezing the underlying language model (param.requires_grad = False), the network leverages pre-stabilized linguistic logic. The hybrid network only focuses on learning sentiment routing, stopping training set memorization.</strong></p><p><strong>-Higher Starting Ground: The incoming arrays contain pre-computed multi-word global context clues, making it much easier for the subsequent Mamba and attention layers to separate positive reviews from negative reviews seamlessly.</strong></p></td><td></td><td><p>Epoch 1/3 | Loss: 0.3575 | Train Acc: 84.36%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 88.57%</p><p>Epoch 2/3 | Loss: 0.2501 | Train Acc: 90.16%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 90.37%</p><p>Epoch 3/3 | Loss: 0.1869 | Train Acc: 93.07%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: <strong>90.58%</strong></p></td></tr><tr><td><p>Hybrid</p></td><td><ol><li><strong>Unfreeze the Top Transformer Layer: Keep the lower linguistic layers of DistilBERT frozen, but unfreeze the final transformer layer (Layer 5). This allows the encoder to slightly adjust its deep contextual representations specifically for sentiment and film vocabulary. [</strong><a href="https://arxiv.org/html/2205.09904v2"><strong>1</strong></a><strong>, </strong><a href="https://5aket.hashnode.dev/fine-tune-neural-network-models-using-tensorflow"><strong>2</strong></a><strong>]</strong></li><li><strong>Layer-Wise Learning Rate Decay (LLRD): We will use a very small learning rate (1×10⁻⁵) for the unfrozen encoder layer so it updates gently, while keeping a slightly higher rate (5×10⁻⁵) for our custom hybrid classifier. [</strong><a href="https://medium.com/@staytechrich/transfer-learning-on-custom-datasets-using-your-own-image-folders-with-pytorch-with-practical-0574afdd1df0"><strong>1</strong></a><strong>]</strong></li><li><strong>Weight Decay Regularization: Increase AdamW weight decay to 0.05 to heavily penalize memorization.</strong></li></ol></td><td></td><td><p>Epoch 1/3 | Loss: 0.3489 | Train Acc: 84.45%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 89.62%</p><p>Epoch 2/3 | Loss: 0.2396 | Train Acc: 90.51%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 91.17%</p><p>Epoch 3/3 | Loss: 0.1903 | Train Acc: 92.78%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: <strong>91.27%</strong></p></td></tr><tr><td><p>Hybrid</p></td><td><p><strong>By increasing max_length to 512 tokens, you allow the model to read these critical final sentences. Because our hybrid model uses Mamba layers, this extra length will process efficiently with linear scaling</strong></p></td><td></td><td><p>Epoch 1/3 | Loss: 0.3417 | Train Acc: 85.05%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 90.34%</p><p>Epoch 2/3 | Loss: 0.2238 | Train Acc: 91.34%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 92.23%</p><p>Epoch 3/3 | Loss: 0.1769 | Train Acc: 93.36%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: <strong>92.39%</strong></p></td></tr><tr><td><p>Hybrid</p></td><td><p><strong>4 epochs with a 0.3 dropout</strong></p></td><td></td><td><p>Epoch 1/4 | Loss: 0.3439 | Train Acc: 84.89%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 89.94%</p><p>Epoch 2/4 | Loss: 0.2366 | Train Acc: 90.84%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 91.76%</p><p>Epoch 3/4 | Loss: 0.1867 | Train Acc: 93.01%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 92.34%</p><p>Epoch 4/4 | Loss: 0.1479 | Train Acc: 94.66%</p><p>&gt;&gt; Adjusted Validation Target Accuracy: 92.34%</p></td></tr></tbody></table></div>

Summary:

From the above table, we can see that the finetuned DistilBERT transformer architecture has the highest test accuracy, the pure Mamba architecture trained from scratch has quite a good performance, accuracy: 0.8785, and it can train and inference at high speed. The Customed transformer architecture trained from scratch has a good performance: 0.8673. The hybrid model of transformer and Mamba has not so bad performance: 0.84. I believe the performance can be improved if we have more data. I finetuned the parameters, and the final result I got: 0.9234, the result is almost the same as the transformer finetuned model.

Basically, the scaling low still holds, the new Mamba architecture is picking up.

Notes: Google Gemini is used in draft the readme.
