# The Complete AI Engineer Roadmap: Beginner to Advanced

This roadmap takes you from programming fundamentals to **Machine Learning (ML), Artificial Neural Networks (ANNs), Deep Learning, CNNs, Transformers, LLMs, RAG, AI agents, and production-grade AI systems**.

The goal is not just to learn AI theory. It is to learn how to build, train, evaluate, deploy, and maintain real AI applications.

## Roadmap overview

1. Programming and computer science
2. Mathematics and data science
3. Classical machine learning
4. Deep learning: ANNs, backpropagation and optimization
5. CNNs and computer vision
6. NLP, attention and Transformers
7. LLM engineering and fine-tuning
8. RAG and knowledge systems
9. AI agents and agentic workflows
10. Production AI engineering and MLOps
11. Advanced specializations

**Important distinction:** ML, neural networks, CNNs, and LLMs are not completely separate subjects. Neural networks are one family of ML models, CNNs are a particular neural-network architecture, and modern LLMs primarily use Transformer architectures. Agents are systems that combine models with tools, state, and controlled execution.

## 1. Complete curriculum at a glance

The following time estimates assume approximately 10–15 hours of study per week, including coding and projects.

| Phase | Subject | Approximate duration |
|---|---|---:|
| 0 | Computer and programming foundations | 3–5 weeks |
| 1 | Python for data and mathematics | 3–5 weeks |
| 2 | Mathematics for ML and deep learning | 4–8 weeks |
| 3 | Data science and exploratory analysis | 3–4 weeks |
| 4 | Classical machine learning | 6–8 weeks |
| 5 | ANN and deep learning fundamentals | 5–7 weeks |
| 6 | CNN and computer vision | 4–6 weeks |
| 7 | NLP, attention and Transformers | 4–6 weeks |
| 8 | LLM engineering and fine-tuning | 5–8 weeks |
| 9 | RAG and knowledge systems | 3–5 weeks |
| 10 | AI agents and tool orchestration | 4–6 weeks |
| 11 | Deployment, MLOps and production | 5–8 weeks |
| 12 | Advanced specializations | Ongoing |

Some phases overlap, so do not simply add every minimum and maximum. A practical target is approximately **9–15 months to become a strong, project-capable AI engineer**, with advanced research, distributed training and production experience requiring further practice.

**Recommended strategy:** learn the foundations in order, but do not wait until you have mastered all the mathematics before building things. Start small projects early, then revisit the theory as the projects become more complex.

---

## 2. Phase 0 — Computer science and programming foundations

**Priority: Essential.** Become comfortable writing, running, debugging and organizing code before serious model training.

### A. Python programming

Learn these concepts in order:

1. Variables, primitive types, operators, strings and input/output.
2. Conditional statements, loops and comprehensions.
3. Lists, tuples, sets and dictionaries.
4. Functions, parameters, return values and scope.
5. Modules, imports and packages.
6. File handling: TXT, CSV, JSON and binary files.
7. Exceptions, debugging and logging.
8. Classes, objects, inheritance, composition and dataclasses.
9. Iterators, generators, decorators and context managers.
10. Type hints, annotations and static checking.
11. List comprehensions, lambda functions and functional programming basics.
12. Virtual environments, dependency management and package creation.
13. Asynchronous Python, `async`/`await`, and concurrency basics.
14. HTTP requests, APIs, JSON responses and environment variables.

Later, learn memory management, profiling, multiprocessing, threading, the Python Global Interpreter Lock, and performance optimization.

### B. Computer science fundamentals

- **Data structures:** arrays, linked lists, stacks, queues, hash tables, trees, graphs and heaps.
- **Algorithms:** searching, sorting, recursion, binary search, graph traversal and dynamic programming basics.
- **Complexity analysis:** Big O time and space.
- **Operating systems:** processes, threads, memory and files.
- **Networking:** HTTP, HTTPS, DNS, TCP/IP and REST APIs.
- **Databases:** relational tables, keys, indexes, joins and transactions.
- **Basic security:** authentication, authorization, secrets and permissions.

You do not need to master competitive programming, but you should be able to manipulate data, reason about performance, and debug programs.

### C. Tools to install and learn

- **Editor:** [Visual Studio Code](https://code.visualstudio.com/) or PyCharm. Learn the debugger, integrated terminal, extensions, code navigation and formatting.
- **Programming language:** [Python](https://docs.python.org/3/tutorial/). Use a version supported by your chosen libraries.
- **Version control:** [Git](https://git-scm.com/doc/) and [GitHub](https://docs.github.com/). Learn commits, branches, merges, pull requests, `.gitignore` and conflict resolution.
- **Interactive experimentation:** [Jupyter](https://jupyter.org/) for notebooks; [Google Colab](https://colab.research.google.com/) for cloud experiments without managing local GPU hardware.
- **Additional tools:** Linux or macOS terminal, Windows PowerShell, Bash basics, `uv` or `pip` for packages, `venv` for virtual environments, `pytest` for testing, and `ruff` for linting and formatting.
- Learn Docker later, but become familiar with a terminal from the beginning.

### D. Beginner projects

- A command-line calculator and unit converter.
- A file organizer that handles errors safely.
- A program that fetches data from a public API and stores it in JSON or CSV.
- A Python package with a few functions, unit tests and a GitHub repository.

**Exit criterion:** You can build a small Python application without copying every line from a tutorial, debug it, test it, and explain how its code is organized.

---

## 3. Phase 1 — Mathematics for machine learning

You do not need a mathematics degree. You do need enough mathematics to understand why a model works, when it fails, and which modifications are appropriate.

### A. Linear algebra

Learn:

- Scalars, vectors, matrices and tensors.
- Vector addition, dot products and matrix multiplication.
- Matrix shapes and dimensionality.
- Transpose, inverse and identity matrices.
- Linear transformations and basis changes.
- Linear independence, rank and orthogonality.
- Eigenvalues and eigenvectors.
- Singular value decomposition (SVD).
- Norms, distances and cosine similarity.
- Positive-definite matrices and covariance matrices at an introductory level.

**Why it matters:** neural networks repeatedly transform vectors and matrices. Embeddings, PCA, attention, image features and optimization all rely on linear algebra.

**Tools:** NumPy, SymPy for symbolic mathematics, and Matplotlib for plotting.

### B. Calculus and optimization

Learn:

- Functions and their graphs.
- Limits and continuity basics.
- Derivatives and partial derivatives.
- Chain rule.
- Gradients and directional derivatives.
- Multivariable calculus.
- Gradient descent.
- Learning rates and convergence.
- Local minima, saddle points and loss surfaces.
- Optimization with constraints at a conceptual level.

The critical principle is that a model's parameters are adjusted using information about how its loss changes:

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
\]

Here, `θ` represents model parameters, `L` is the loss function, and `η` is the learning rate.

### C. Probability and statistics

Learn:

- Events, conditional probability and independence.
- Bayes' theorem.
- Random variables and probability distributions.
- Expectation, variance and standard deviation.
- Bernoulli, binomial, Gaussian and categorical distributions.
- Sampling, bias and variance.
- Correlation versus causation.
- Confidence intervals and hypothesis testing.
- Maximum likelihood estimation.
- Sampling distributions and statistical significance.
- Calibration and uncertainty.
- Bayesian reasoning basics.

### D. Information theory

Especially useful for deep learning and LLMs:

- Entropy.
- Cross-entropy.
- Kullback–Leibler (KL) divergence.
- Information gain.
- Perplexity.
- Mutual information at an introductory level.

Cross-entropy is particularly important for understanding classification and next-token prediction.

### E. Additional mathematics

Eventually study numerical computing, logarithms and exponentials, discrete mathematics, graph theory, optimization theory, and signal-processing fundamentals for computer vision and audio.

**Best learning approach:** visualize vectors, derivatives, probability distributions and gradient descent in Python. A concept you can implement and plot is usually easier to remember than one you only read.

---

## 4. Phase 2 — Data science and data engineering fundamentals

Machine learning performance depends heavily on data quality. Learn to prepare data before trying to build complex models.

### Concepts to master

- **Data manipulation:** loading, filtering, joining, sorting, grouping, aggregation, missing-value handling, duplicate removal and type conversion.
- **Exploratory data analysis (EDA):** distributions, correlations, outliers, class imbalance, feature relationships and visualizations.
- **Data preparation:** numerical scaling, categorical encoding, text cleaning, image preprocessing, normalization and feature extraction.
- **Data splitting:** training, validation and test sets, stratified sampling, grouped splitting and time-aware splitting.
- **Data quality:** schema validation, data contracts, provenance, versioning and reproducibility.
- **Data leakage:** recognizing when training data accidentally contains information that would not be available at prediction time.

### Tool list

| Tool | What to learn |
|---|---|
| NumPy | Arrays, vectorization, broadcasting, indexing and linear algebra |
| pandas | DataFrames, joins, groupby, missing data and time series |
| Matplotlib | Figures, plots, histograms and visual diagnostics |
| Seaborn | Statistical visualizations |
| SciPy | Scientific computing, statistics and optimization |
| Statsmodels | Statistical modeling, diagnostics and interpretable statistical analysis |
| SQL | `SELECT`, `JOIN`, `GROUP BY`, CTEs, window functions and indexes |
| PostgreSQL | Relational database operations and persistence |
| DuckDB | Local analytical SQL over files and datasets |
| Jupyter | Exploratory analysis and reproducible experiments |
| Parquet and Apache Arrow | Efficient analytical data storage and columnar formats |

Optional for larger datasets: Polars, PyArrow, Spark, and cloud data warehouses such as BigQuery, Snowflake or Redshift.

### Project

Take a real dataset, document its quality, perform EDA, clean it, construct a reliable train/validation/test split, and produce a report with visualizations.

Possible datasets: [Kaggle](https://www.kaggle.com/datasets), [UCI Machine Learning Repository](https://archive.ics.uci.edu/), and [Hugging Face Datasets](https://huggingface.co/datasets).

**Exit criterion:** You can take an unfamiliar dataset and produce a reproducible, well-documented dataset ready for training without leaking information across splits.

---

## 5. Phase 3 — Classical machine learning

This phase teaches you how models learn from examples. It also prevents a common mistake: assuming neural networks are always the best solution.

A good starting resource is the [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/), which covers regression, classification, data, neural networks, embeddings, LLM fundamentals and production ML.

### A. Supervised learning

| Algorithm | Key concepts |
|---|---|
| Linear regression | Least squares, residuals, regularization |
| Logistic regression | Probabilities, decision thresholds, log loss |
| k-nearest neighbors | Distance metrics and neighborhood size |
| Naive Bayes | Conditional independence and probability |
| Decision trees | Splits, entropy, Gini impurity, pruning |
| Random forests | Bagging, feature subsampling and variance reduction |
| Gradient boosting | Sequential error correction and weak learners |
| XGBoost, LightGBM and CatBoost | Modern gradient-boosted decision trees |
| Support vector machines | Margins, kernels and support vectors |
| Linear and kernel methods | High-dimensional features and regularization |

Be able to explain the assumptions, strengths, limitations and computational trade-offs of the algorithms you use.

### B. Unsupervised and other learning methods

- k-means clustering.
- Hierarchical clustering.
- DBSCAN.
- Gaussian mixture models.
- Principal component analysis (PCA).
- Dimensionality reduction with t-SNE and UMAP.
- Anomaly detection.
- Association rules at a basic level.
- Semi-supervised learning.
- Self-supervised learning.
- Introductory reinforcement learning: states, actions, rewards, policies and value functions.

You do not need to master reinforcement learning before learning LLMs, but it becomes relevant for advanced model training and agent research.

### C. Model evaluation

For regression, learn MAE, MSE, RMSE and `R²`.

For classification, learn:

- Accuracy and balanced accuracy.
- Precision and recall.
- F1 score.
- Confusion matrices.
- ROC-AUC and PR-AUC.
- Log loss.
- Probability calibration.
- Threshold selection.
- Cost-sensitive metrics.

For all models, study cross-validation, learning curves, bias versus variance, underfitting, overfitting, class imbalance, data leakage, hyperparameter tuning and baseline comparisons.

### D. Scikit-learn toolbox

Master the most useful parts of [scikit-learn](https://scikit-learn.org/stable/user_guide.html):

- `train_test_split`
- `Pipeline`
- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`
- `SimpleImputer`
- Model-specific estimators.
- Cross-validation and hyperparameter search.
- Classification and regression metrics.
- Feature selection and permutation importance.
- Model persistence and reproducibility.

### E. Advanced ML topics

After the basics, learn:

- Regularization: L1, L2 and elastic net.
- Hyperparameter optimization: grid search, random search and Bayesian optimization.
- Feature selection and feature importance.
- Class weighting and resampling.
- Interpretability: SHAP and permutation importance.
- Calibration, uncertainty and decision thresholds.
- Fairness, subgroup analysis and robustness.
- Time-series forecasting and leakage-safe temporal validation.
- Recommender systems and ranking.
- Distribution shift and concept drift.

Tools to add: Optuna for tuning, XGBoost, LightGBM, CatBoost and SHAP.

### Projects

Build at least three:

1. A house-price or demand prediction model.
2. A fraud detection or customer-churn classifier.
3. A clustering or anomaly-detection system.

For each project, establish a simple baseline, evaluate it correctly, compare models, tune parameters, analyze failure cases, and write down what you learned.

**Exit criterion:** You can select and evaluate a model based on a real problem, not simply choose the most complicated available algorithm.

---

## 6. Phase 4 — Artificial Neural Networks (ANNs) and deep learning

This is the bridge between classical ML and CNNs, Transformers and LLMs.

An artificial neural network learns a mapping from inputs to outputs by adjusting trainable parameters to reduce a loss function.

### A. Understand how one neuron works

A basic neuron computes:

\[
z=\mathbf{w}^{T}\mathbf{x}+b
\]

\[
a=f(z)
\]

Where:

- `x`: input features.
- `w`: learnable weights.
- `b`: bias.
- `f`: activation function.
- `a`: neuron's output.

A multilayer network combines many such operations.

### B. ANN curriculum

Learn each subject in this order:

1. Biological inspiration versus mathematical artificial neurons.
2. Perceptron and linear decision boundaries.
3. Input, hidden and output layers.
4. Feed-forward neural networks and multilayer perceptrons (MLPs).
5. Activation functions: ReLU, sigmoid, tanh, GELU and SiLU.
6. Output layers: softmax, sigmoid and linear output.
7. Loss functions: mean squared error, cross-entropy and binary cross-entropy.
8. Forward propagation.
9. Backpropagation and the chain rule.
10. Computational graphs and automatic differentiation.
11. Gradient descent, stochastic gradient descent and mini-batch learning.
12. Optimizers: SGD with momentum, Adam and AdamW.
13. Learning rates and learning-rate schedules.
14. Batch size, epochs and iterations.
15. Weight initialization: Xavier/Glorot and He initialization.
16. Overfitting, underfitting and generalization.
17. Dropout, weight decay and early stopping.
18. Batch normalization and layer normalization.
19. Vanishing and exploding gradients.
20. Gradient clipping and numerical stability.
21. Hyperparameter tuning, checkpoints and reproducible training.

### C. The framework to learn: PyTorch

Use the [PyTorch official tutorials](https://docs.pytorch.org/tutorials/) as your primary practical guide.

PyTorch tools checklist:

- [ ] Tensor creation, shapes, indexing and broadcasting.
- [ ] CPU/GPU devices and tensor movement.
- [ ] `torch.nn.Module` and model layers.
- [ ] `torch.autograd` and gradient computation.
- [ ] `Dataset` and `DataLoader`.
- [ ] Loss functions and optimizers.
- [ ] Custom training and validation loops.
- [ ] Evaluation mode versus training mode.
- [ ] Saving checkpoints and restoring models.
- [ ] GPU memory management and profiling.
- [ ] Automatic mixed precision.
- [ ] TensorBoard or another experiment visualization tool.

TensorFlow and Keras are useful alternatives, particularly in projects and organizations built around their ecosystem. You do not need to learn every deep-learning framework simultaneously.

### D. ANN projects

**Project 1 — Build an ANN from scratch.** Use NumPy to implement a forward pass, loss computation, backpropagation and gradient descent for a small network. Do not use PyTorch autograd for this exercise.

**Project 2 — Handwritten digit classification.** Train an MLP on MNIST, examine the loss curves, and compare different optimizers and learning rates.

**Project 3 — Tabular neural network.** Predict a numeric target or classify customer behavior. Compare the ANN against a strong scikit-learn or gradient-boosted-tree baseline.

**Exit criterion:** You can derive the basic gradient equations, implement a training loop, diagnose overfitting, and explain why a model's loss is or is not improving.

---

## 7. Phase 5 — CNNs and computer vision

Convolutional Neural Networks are especially useful for learning spatial patterns in images. This branch also prepares you for image segmentation, object detection and some multimodal AI systems.

### A. CNN concepts in the proper order

| Topic | What to understand |
|---|---|
| Images as tensors | Height, width, channels, color representation |
| Convolution | Kernels, filters, feature maps |
| Stride and padding | Output dimensions and spatial resolution |
| Channels | Input and output feature channels |
| Receptive fields | How much of the input influences a feature |
| Pooling | Max pooling and average pooling |
| Activation and normalization | ReLU, batch normalization and alternatives |
| Downsampling | Trade-offs between resolution and computation |
| Residual connections | Skip connections and deeper networks |
| Transfer learning | Reusing pretrained representations |
| Data augmentation | Cropping, flipping, rotation and color transformations |

Understand the output dimensions of convolution layers mathematically, not only through experimentation.

### B. CNN architectures to study

Learn the historical progression and why later designs improved on earlier ones:

- **LeNet:** a foundational convolutional architecture.
- **AlexNet:** large-scale deep CNN classification.
- **VGG:** straightforward stacks of convolution layers.
- **Inception/GoogLeNet:** parallel convolution operations.
- **ResNet:** residual connections and deeper networks.
- **DenseNet:** dense feature connections.
- **EfficientNet:** systematic scaling of network dimensions.
- **ConvNeXt:** modernized convolutional design.

After CNNs, learn Vision Transformers (ViTs), patch embeddings, image attention and hybrid CNN–Transformer architectures.

### C. Computer vision tasks

- Image classification.
- Multi-label classification.
- Object detection and bounding boxes.
- Semantic segmentation.
- Instance segmentation.
- Image similarity and metric learning.
- Face and object embeddings, where appropriate.
- Image generation and diffusion models as an advanced branch.

### D. Essential tools

| Tool | Purpose |
|---|---|
| PyTorch | Training neural networks |
| `torchvision` | Common vision datasets, models and transforms |
| `timm` | Broad collection of image-model architectures |
| Pillow | Basic image loading and manipulation |
| OpenCV | Image and video processing |
| Albumentations | Advanced image augmentation |
| CVAT or Label Studio | Data annotation |
| Ultralytics YOLO or Detectron2 | Selected object-detection workflows |
| TorchMetrics | Standardized metrics |
| Weights & Biases or TensorBoard | Experiment visualization |

You do not have to learn all the vision libraries at the same time. Start with PyTorch and `torchvision`, then add tools according to your project.

### E. Projects

1. Classify images across several categories using a CNN.
2. Fine-tune a pretrained ResNet or similar architecture on a small custom dataset.
3. Build an object detector that finds objects in images.
4. For advanced work, implement image segmentation and measure intersection over union (IoU).

For serious vision work, know the differences between accuracy, precision, recall, mAP, IoU and Dice score.

**Exit criterion:** You can train a CNN, use a pretrained model, inspect prediction errors, select appropriate evaluation metrics and deploy an image inference endpoint.

---

## 8. Phase 6 — NLP, attention and Transformers

This phase explains the technical foundations that lead to LLMs.

### A. Natural Language Processing foundations

Learn traditional NLP and the transition to neural methods:

1. Text normalization and preprocessing.
2. Sentence splitting and tokenization.
3. Word, character and subword representations.
4. Bag-of-words and TF-IDF.
5. Word embeddings: Word2Vec, GloVe and FastText.
6. Sequence modeling with RNNs.
7. LSTM and GRU architectures.
8. Sequence-to-sequence models.
9. Attention mechanisms.
10. Transformer architecture.

Learn how tokenization changes text into integer token IDs, and how padding, truncation, vocabulary size and special tokens affect the model.

### B. Transformer architecture

Study these topics carefully:

- Query, key and value vectors: Q, K and V.
- Scaled dot-product attention.
- Multi-head attention.
- Positional encodings and positional embeddings.
- Causal attention masks.
- Encoder, decoder and encoder–decoder designs.
- Residual connections.
- Layer normalization.
- Feed-forward sublayers.
- Token embeddings and output projections.
- Self-attention complexity and memory consumption.
- Autoregressive generation.
- Decoding strategies: greedy decoding, beam search and sampling.

The scaled dot-product attention mechanism is commonly expressed as:

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
\]

Here, `M` represents an optional attention mask. It is particularly important in decoder-only language models because causal masking prevents each token from attending to future tokens.

### C. Major model families

| Model family | Primary learning purpose |
|---|---|
| BERT and encoder-only models | Classification, representations and masked language modeling |
| GPT-style decoder-only models | Autoregressive text generation |
| T5 and encoder–decoder models | Text-to-text tasks, translation and summarization |
| Vision Transformers | Applying attention to images |
| Encoder–decoder speech models | Speech recognition and related audio tasks |

Do not memorize model names without learning the architectural differences.

### D. Practical exercise

Implement a small self-attention module and then a miniature Transformer in PyTorch. Train it on a small text dataset or character-level sequence prediction task.

This exercise should teach you tensor shapes, masks, attention, token embeddings, loss computation and autoregressive generation. Afterward, use pretrained models through Hugging Face rather than trying to recreate every component for every practical application.

---

## 9. Phase 7 — Large Language Models (LLMs)

**Goal:** Move from understanding neural networks to building, customizing and serving language models.

### A. LLM fundamentals

#### Tokenization and context

- Vocabulary, token IDs and special tokens.
- Chat templates and message roles.
- Context windows and context limits.
- Token counts, truncation and padding.
- System, user and assistant messages.
- Why the same text can be represented by different tokenizers.

#### Pretraining

- Next-token prediction.
- Training corpora and data preprocessing.
- Cross-entropy loss and perplexity.
- Training and validation loss.
- Scaling in model parameters, data and compute.
- Data quality, contamination and duplication.
- Why pretraining does not guarantee factual knowledge or reliable behavior.

#### Inference

- Temperature, top-k and top-p sampling.
- Greedy decoding and beam search.
- Stop conditions.
- Structured output and schema validation.
- Streaming, batching and context management.
- KV caching and inference cost.

#### Post-training

- Supervised fine-tuning (SFT).
- Instruction tuning and chat datasets.
- Preference data and reward models.
- RLHF: reinforcement learning from human feedback.
- DPO: direct preference optimization.
- Tool-use and function-calling data.
- Safety tuning and behavioral evaluations.

You should be able to distinguish a model that has been pretrained, one fine-tuned for a particular task, and a complete application that uses a model with retrieval and tools.

### B. Essential Hugging Face ecosystem

The [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1) is a practical resource for Transformers, datasets, tokenizers and model workflows.

| Library or tool | What it does |
|---|---|
| `transformers` | Loads and uses pretrained Transformer models |
| `datasets` | Loads, processes and organizes datasets |
| `tokenizers` | Builds and operates efficient tokenizers |
| `evaluate` | Computes model evaluation metrics |
| `accelerate` | Simplifies device placement and distributed training workflows |
| `peft` | Parameter-efficient fine-tuning, including LoRA |
| `trl` | Training techniques for language models, including SFT and preference optimization |
| `safetensors` | Safe, efficient tensor serialization |
| `bitsandbytes` | Quantization and memory-efficient operations in supported workflows |
| Hugging Face Hub | Publishes, versions and discovers models and datasets |

Learn the basics of `transformers` and `datasets` first. Add the others when you reach fine-tuning or deployment.

### C. API-based models versus open-weight models

Learn both approaches, but start with an API or a small pretrained model.

**Hosted model APIs:** examples include OpenAI, Anthropic and Google Gemini. Understand authentication, rate limits, token costs, latency, structured responses and provider-specific capabilities.

**Open-weight models:** use the Hugging Face Hub to explore compatible models, licenses and hardware needs.

For local inference and serving, learn the difference between:

- **Ollama:** convenient local model workflows.
- **llama.cpp:** efficient local inference for supported model formats, especially quantized models.
- **vLLM:** high-throughput model serving.
- **Hugging Face Text Generation Inference:** an alternative model-serving stack.

You do not need to use all four. Select a tool based on model compatibility, deployment environment and throughput requirements. See the [vLLM OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

### D. Fine-tuning and advanced model customization

Learn these techniques in increasing order of complexity:

1. Fine-tune a pretrained model for text classification.
2. Perform supervised instruction fine-tuning.
3. Learn parameter-efficient fine-tuning (PEFT).
4. Implement LoRA.
5. Explore QLoRA and low-bit quantization.
6. Understand full fine-tuning versus adapter fine-tuning.
7. Explore preference optimization such as DPO.
8. Study RLHF, reward models and policy optimization conceptually.
9. Learn distributed training with tools such as PyTorch FSDP, DeepSpeed and Hugging Face Accelerate.
10. Learn data deduplication, checkpointing, mixed-precision training, gradient accumulation and training-memory optimization.

**Important:** do not begin by attempting to train a frontier-scale LLM from scratch. Train a tiny Transformer for understanding, fine-tune a small pretrained model, and then expand only when you have a clear need and adequate compute.

### E. LLM evaluation

A model that produces fluent text is not necessarily a useful or reliable model. Learn to measure:

- Exact match and task-specific accuracy.
- Human preference and human-reviewed quality.
- Instruction following and structured-output validity.
- Factual correctness and groundedness.
- Hallucination rates on a defined test set.
- Latency, throughput and token costs.
- Performance across different user groups and input types.
- Robustness and appropriate handling of adversarial or out-of-distribution inputs.

For language modeling, understand perplexity; for specific tasks, choose relevant metrics rather than relying on one universal score.

---

## 10. Phase 8 — RAG and enterprise knowledge systems

Retrieval-Augmented Generation (RAG) gives an LLM access to relevant external information at response time. It is especially useful when applications need private documents, frequently changing information, source references or searchable knowledge bases.

### A. RAG data flow

1. **Data ingestion:** PDFs, HTML, Markdown, databases and APIs.
2. **Parsing and chunking:** extract text, split logically and attach metadata.
3. **Embedding and indexing:** embedding model plus searchable index.
4. **Retrieval:** semantic search, keyword search, filters and reranking.
5. **Generation:** give the retrieved context to the LLM.
6. **Validation and evaluation:** verify evidence, answer quality, access control and citations.

### B. RAG concepts to master

- Document extraction and parsing.
- Chunk size, overlap and semantic chunking.
- Metadata and document identifiers.
- Dense embeddings and similarity metrics.
- Keyword search using BM25.
- Hybrid retrieval combining lexical and semantic search.
- Query rewriting and query expansion.
- Metadata filters.
- Reranking using cross-encoders.
- Context selection and context-window budgets.
- Citation generation and source attribution.
- Retrieval quality versus answer quality.
- Incremental indexing, updates and deletions.
- Document permissions and tenant isolation.

### C. Vector search tools

| Tool | When to study it |
|---|---|
| FAISS | Local similarity search and index experimentation |
| PostgreSQL with `pgvector` | SQL applications that need vector search |
| Qdrant | Dedicated vector search service |
| Weaviate | Vector search and knowledge retrieval applications |
| Milvus | Larger-scale vector database workflows |
| Pinecone | Managed vector database workflows |

These are alternatives, not a requirement to learn every database. Start with FAISS or `pgvector`, then choose a production option based on your data, hosting model and scale.

For embeddings, study Sentence Transformers and relevant embedding models from the Hugging Face Hub or hosted model providers.

### D. Frameworks

Study either [LangChain](https://www.langchain.com/) or [LlamaIndex](https://www.llamaindex.ai/) after building a basic retrieval pipeline yourself.

Learn what the framework is doing underneath it: document ingestion, splitting, indexing, retrieval, prompting and generation. Hiding these operations behind a framework makes debugging poor retrieval more difficult.

### E. RAG evaluation

Measure retrieval and generation separately.

**Retrieval:** Recall@k, Precision@k, Mean Reciprocal Rank (MRR), and normalized Discounted Cumulative Gain (nDCG).

**Generation:** groundedness, answer correctness, completeness, citation correctness and appropriate refusal when evidence is insufficient.

### F. Project

Build a document-question-answering system for a collection of manuals or technical documents.

The system must return answers with source references, identify when the documents do not contain the answer, respect access restrictions, and have a test set of known questions and expected evidence.

Then compare different chunk sizes, embedding models, retrievers and rerankers.

**Exit criterion:** You can tell whether poor answers originate from document parsing, retrieval, insufficient context, or generation—and improve the correct stage.

---

## 11. Phase 9 — AI agents, tool use and agentic workflows

An LLM application is not automatically an agent. An agentic system generally uses a model to choose or coordinate actions through tools, observe results, and continue toward a goal within defined constraints.

The [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction) covers agent fundamentals, frameworks, practical use cases, evaluation and deployment.

### A. Architecture of an agent

1. **Model:** interprets the task and decides the next step.
2. **Instructions:** defines the task, boundaries and expected behavior.
3. **Tools:** functions, APIs, search, databases and other permitted actions.
4. **Control loop:** interprets a response, executes a permitted action, observes the result and continues or stops.
5. **State:** records progress through the current task.
6. **Memory:** stores relevant information across steps or conversations.
7. **Planning:** decomposes a request into smaller tasks when needed.
8. **Validation:** checks tool inputs, outputs and completion conditions.
9. **Guardrails:** limits unsafe or unauthorized behavior.
10. **Observability:** records executions, tool calls, errors and cost.
11. **Evaluation:** tests whether the system actually completes tasks correctly.

A common workflow is:

**Observe → Decide → Act → Observe again**

Real implementations need explicit stopping conditions, error handling and limits on the number of actions.

### B. Agent concepts: beginner to advanced

**Level 1 — Basic tool use**

- Function calling and JSON schemas.
- Input validation.
- Calling a calculator, search API or database query.
- Parsing tool results and returning an answer.

**Level 2 — Stateful workflows**

- Multi-turn conversation state.
- Workflow steps and branching.
- Retries, timeouts and error recovery.
- Structured state management.
- Saving intermediate results.

**Level 3 — RAG and specialized tools**

- Retrieval tools.
- Search and research tools.
- SQL and analytical tools.
- File processing.
- Calling other services through APIs.
- Human approval for consequential actions.

**Level 4 — Advanced agents**

- Planning and task decomposition.
- Dynamic tool selection.
- Long-running workflows.
- Persistent memory and checkpoints.
- Parallel execution where appropriate.
- Multi-agent delegation.
- Sandboxed code execution.
- Evaluation and observability.
- Cost and execution-budget management.

**Level 5 — Production-grade agent systems**

- Permissioned tool access.
- User-specific authorization.
- Secrets management.
- Recovery after crashes.
- Idempotent operations.
- Audit logs.
- Versioned tools and prompts.
- Human-in-the-loop approval.
- Isolation of untrusted content.
- Adversarial testing and security monitoring.

### C. Agent tool ecosystem

| Tool or framework | Purpose |
|---|---|
| Direct model APIs with function calling | Learn the core tool-use mechanism |
| OpenAI Agents SDK | Build code-first agents and orchestrated tool workflows |
| LangGraph | Build stateful, graph-based workflows and agents |
| Hugging Face smolagents | Learn a lightweight agent framework |
| LlamaIndex | Integrate data retrieval and agent workflows |
| Microsoft Agent Framework | An alternative for Microsoft-oriented agent development |
| MCP (Model Context Protocol) | Standardized way to expose tools and context to compatible clients |
| Pydantic | Validate structured inputs and outputs |
| MLflow Tracing or another tracing solution | Inspect agent executions and troubleshoot failures |

Relevant documentation: [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents/sdk) and [LangGraph durable execution](https://langchain-ai.github.io/langgraph/concepts/durable_execution/).

**Important rule:** choose one agent framework initially. Learn direct tool calling first so you understand the underlying mechanism. Build multi-agent systems only when a single-agent design or deterministic workflow is no longer sufficient.

### D. Agent projects

1. A tool-using assistant that calls a calculator and a public API.
2. A document-research agent that retrieves sources before answering.
3. A data analyst that queries a read-only database and explains results.
4. A multi-step workflow that saves progress and resumes after an error.
5. A business-process agent requiring human approval before performing consequential operations.
6. An advanced system that coordinates specialized agents, with evaluation tests for each agent and the overall system.

### E. Agent security

Study prompt injection, indirect prompt injection through retrieved documents, unauthorized tool calls, data exfiltration, insecure output handling, resource exhaustion, excessive agency and insecure code execution.

Learn to enforce permissions outside the model, validate all tool arguments, constrain database access, sandbox code, limit tool execution, protect secrets, log sensitive actions appropriately and request approval before consequential operations.

The [OWASP GenAI Security Project](https://genai.owasp.org/) publishes risks and mitigations for LLM and agentic applications.

**Exit criterion:** you can explain every action an agent is permitted to take, demonstrate how failures are recovered, and show evidence that the agent is tested against both normal and adversarial scenarios.

---

## 12. Phase 10 — Production AI engineering and MLOps

Training a model or building a working RAG system is only part of AI engineering. You also need to package, deploy, secure, monitor and update it.

### A. Software and API engineering

Learn:

- Clean project structure and separation of responsibilities.
- Configuration management.
- Type hints and schema validation.
- Unit, integration and end-to-end testing.
- REST API design.
- Authentication and authorization.
- Asynchronous request handling.
- Database connections and transactions.
- Logging, error responses and health checks.
- API versioning.
- CI/CD and automated testing.

**Tools:** FastAPI, Pydantic, pytest, GitHub Actions, PostgreSQL and a database migration tool such as Alembic.

The [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) covers API construction, validation and progressively more advanced application features.

### B. Containers and deployment

Study these tools in order:

1. Docker images, containers and Dockerfiles.
2. Volumes, networking, environment variables and secrets.
3. Docker Compose for local multi-service applications.
4. A cloud platform: AWS, Google Cloud or Azure.
5. Container registries and managed deployment services.
6. HTTPS, DNS, reverse proxies and load balancing.
7. Kubernetes when your deployment scale or operational needs justify it.
8. Infrastructure as code, for example Terraform, when managing larger deployments.

Do not attempt to master three cloud platforms simultaneously. Pick one, deploy a small application, and learn its storage, compute, networking, IAM and monitoring services. See [Docker Getting Started](https://docs.docker.com/get-started/).

### C. Experiment tracking and model lifecycle

Understand why experiments must be reproducible.

Learn:

- Logging parameters, metrics and artifacts.
- Dataset and model versioning.
- Reproducing training runs.
- Tracking model configurations.
- Model registries and staged releases.
- Evaluation before promotion.
- Rollback and model comparison.
- Monitoring real-world predictions.

Tools:

- **MLflow:** experiment tracking, model lifecycle, LLM traces and evaluation.
- **Weights & Biases:** experiment tracking and visualization.
- **DVC:** data and model versioning workflows.
- **TensorBoard:** training visualizations.
- **Git and GitHub:** source-code history and collaboration.

You do not need all of these. Start with Git, MLflow and a reproducible experiment structure. Add DVC or Weights & Biases if they solve a real need.

See the [MLflow documentation](https://mlflow.org/docs/latest/).

### D. Production monitoring

For conventional ML, monitor:

- Prediction quality as labels become available.
- Input-data quality and distribution shift.
- Concept drift.
- Latency, throughput and failure rates.
- Model version and data lineage.

For LLMs and agents, additionally monitor:

- Token consumption and model costs.
- Time spent in each workflow stage.
- Retrieval quality and citation correctness.
- Tool-call failures and unnecessary loops.
- Evaluation results by task category.
- Policy violations and security events.
- Human-review rates.

Potential tools include OpenTelemetry, Prometheus, Grafana and MLflow Tracing, or a compatible alternative selected for your environment.

### E. Advanced serving and scalability

Eventually learn:

- GPU memory and utilization.
- Batch inference and continuous batching.
- Quantized inference.
- Streaming generation.
- KV-cache memory.
- Request queues and backpressure.
- Autoscaling and rate limiting.
- Caching strategies.
- Latency versus throughput trade-offs.
- Distributed training and serving.
- Model sharding and parallelism.
- Cost-performance benchmarking.

Relevant technologies include vLLM, `torch.compile`, PyTorch FSDP, DeepSpeed and cloud GPU services. Use them when a measured bottleneck justifies them—not just because they are advanced technologies.

### F. Project

Deploy an ML model behind an API, containerize it, write automated tests, log requests and monitor performance. Then deploy an LLM application with evaluation traces, latency metrics and token-cost tracking.

---

## 13. Advanced specializations after the core roadmap

You do not need to become an expert in every branch of AI. Once you have completed the fundamentals, choose one or two specializations.

### A. Advanced computer vision

Study object detection, segmentation, Vision Transformers, self-supervised vision, image generation, diffusion models, image-text models, multimodal embeddings, video understanding and visual reasoning.

**Tools to explore:** torchvision, timm, OpenCV, Hugging Face Diffusers and selected detection frameworks.

### B. LLM research and model training

Study Transformer internals, distributed training, model scaling, data curation, pretraining objectives, SFT, preference optimization, attention variants, quantization, memory optimization and inference kernels.

**Tools to explore:** PyTorch, Transformers, Accelerate, PEFT, TRL, FSDP and DeepSpeed.

### C. Advanced RAG and search

Study hybrid retrieval, learned reranking, query planning, GraphRAG, knowledge graphs, multimodal retrieval, SQL retrieval, long-context strategies and retrieval evaluation.

**Tools to explore:** Elasticsearch or OpenSearch, Qdrant, pgvector, graph databases and specialized retrieval frameworks.

### D. Advanced agent engineering

Study durable execution, controlled parallelism, task planning, multi-agent delegation, tool reliability, agent benchmarks, sandboxing, memory strategies and evaluation-driven development.

**Tools to explore:** LangGraph, OpenAI Agents SDK, MCP, workflow engines and distributed task queues.

### E. Reinforcement learning and robotics

Study Markov decision processes, dynamic programming, Q-learning, policy gradients, actor-critic methods, PPO, reward design, simulation, control and embodied agents.

**Tools to explore:** Gymnasium, Stable-Baselines3 and selected robotics simulators.

### F. AI infrastructure and platform engineering

Study GPU clusters, distributed systems, inference serving, data pipelines, model registries, resource scheduling, observability and large-scale deployment.

**Tools to explore:** Kubernetes, Ray, vLLM, cloud GPU platforms, MLflow and distributed-training frameworks.

---

## 14. The complete AI tools inventory

This is a practical inventory to work through. It is not literally every tool in the AI industry; there is no fixed exhaustive list, and tools change. Learn the core tools first and treat the other entries as choices for specific needs.

| Category | Core tool or tools | Optional alternatives |
|---|---|---|
| Editor | VS Code | PyCharm |
| Programming | Python | C++ for performance-critical work |
| Package management | `uv`, `pip`, `venv` | Conda |
| Version control | Git, GitHub | GitLab |
| Notebooks | Jupyter | Google Colab, Kaggle Notebooks |
| Math | NumPy, SymPy | SciPy |
| Data processing | pandas | Polars, Spark |
| Visualization | Matplotlib | Seaborn, Plotly |
| Statistics | SciPy, Statsmodels | Specialized statistical packages |
| SQL database | PostgreSQL | MySQL, SQLite |
| Classical ML | scikit-learn | Specialized ML libraries |
| Gradient boosting | XGBoost, LightGBM, CatBoost | Other boosting implementations |
| Deep learning | PyTorch | TensorFlow/Keras |
| Computer vision | torchvision, timm, OpenCV | Selected vision frameworks |
| NLP and LLMs | Transformers, Tokenizers, Datasets | Model-specific toolkits |
| Fine-tuning | PEFT, TRL, Accelerate | Other training pipelines |
| LLM evaluation | Evaluate, custom benchmark suites | Domain-specific evaluation packages |
| Model APIs | OpenAI, Anthropic, Gemini | Other compatible model providers |
| Local inference | Ollama, llama.cpp | Other local runtimes |
| Model serving | vLLM | Other supported serving stacks |
| Embeddings | Sentence Transformers | Hosted embedding APIs |
| Vector search | FAISS, pgvector | Qdrant, Weaviate, Milvus, Pinecone |
| RAG frameworks | LangChain or LlamaIndex | Direct custom pipeline |
| Agent frameworks | LangGraph or OpenAI Agents SDK | smolagents, Microsoft Agent Framework |
| Tool integration | Python functions, REST APIs, MCP | Provider-specific integrations |
| Data validation | Pydantic | JSON Schema libraries |
| Backend APIs | FastAPI | Flask |
| Testing | pytest | unittest |
| Experiment tracking | MLflow | Weights & Biases |
| Data versioning | Git and optional DVC | Other data-versioning workflows |
| Containers | Docker | Podman |
| CI/CD | GitHub Actions | GitLab CI |
| Cloud | AWS, GCP or Azure — choose one | Other cloud platforms |
| Monitoring | OpenTelemetry, Prometheus, Grafana | Other observability platforms |
| Advanced compute | CUDA, FSDP, DeepSpeed | Ray and other distributed systems |
| Security | OWASP guidance, sandboxing, IAM | Organization-specific controls |

### What should you install first?

Resist the temptation to install the entire table.

Start with Python, VS Code, Git, a virtual environment, Jupyter, NumPy, pandas, Matplotlib and scikit-learn. Add PyTorch after you understand basic ML. Add Hugging Face libraries as you learn Transformers and LLMs. Add FastAPI and Docker when you are ready to deploy.

This sequence will keep your environment manageable and reduce time spent debugging tools unrelated to what you are learning.

---

## 15. Your project portfolio: beginner to advanced

These projects should become your evidence of competence. Publish the code, document the experiments and explain the engineering decisions.

1. **Programming and data science:** Build a Python data-processing application, then produce a full EDA report from a real dataset. Evidence: clean code, tests, reusable functions and documented findings.
2. **Classical ML and ANN:** Build a predictive model with a clear baseline, then implement and evaluate a neural network. Evidence: proper data splits, metrics, model comparisons and analysis of errors.
3. **Computer vision:** Fine-tune a pretrained CNN, then optionally add an object detector or segmentation model. Evidence: reproducible training, correct vision metrics and working inference.
4. **Transformer and LLM application:** Implement a small attention-based model, then build an application with a pretrained LLM. Evidence: correct model integration, structured output, evaluation and cost awareness.
5. **Document RAG system:** Ingest a document collection, implement retrieval and reranking, and provide answers with verifiable sources. Evidence: retrieval evaluation, citation correctness and handling of unsupported answers.
6. **Tool-using AI agent:** Build an agent that can search documents, call approved tools, maintain workflow state and request approval before risky operations. Evidence: reliable tool use, recovery, security testing, tracing and measurable task success.
7. **Production deployment:** Serve one of your models or LLM applications through an API, package it in Docker, automate tests and implement monitoring. Evidence: deployment instructions, tests, versioned artifacts, monitoring and a working demo.

For every project, include a README, problem statement, dataset source, setup instructions, architecture, evaluation method, limitations, example outputs and screenshots or a live demonstration where practical.

A portfolio with six thoroughly implemented systems is more persuasive than dozens of copied notebooks.

---

## 16. Best learning resources in the correct order

These resources complement the roadmap without requiring you to buy multiple overlapping courses.

1. **[Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)** — supervised ML, data, classification, neural networks, embeddings, LLM fundamentals and production ML.
2. **[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)** — algorithms, preprocessing, model selection and evaluation in practice.
3. **[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)** — tensors, autograd, neural networks, optimizers, datasets and training loops.
4. **[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)** — Transformers, tokenizers, datasets, pretrained models and fine-tuning.
5. **[Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)** — agent fundamentals, tools, control loops, frameworks and practical applications.
6. **[Docker Getting Started](https://docs.docker.com/get-started/)** — container fundamentals before deploying complex AI systems.
7. **[MLflow Documentation](https://mlflow.org/docs/latest/)** — experiment tracking, model management, evaluation and tracing for GenAI systems.
8. **[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)** — API construction, request validation and production-oriented application patterns.
9. **[OWASP GenAI Security Project](https://genai.owasp.org/)** — risks and mitigations for LLM and agentic applications.

Use official documentation as a technical reference. Implement the exercises instead of only reading them.

---

## 17. A practical weekly study schedule

For a 12-hour study week, use this schedule as a starting point.

| Activity | Hours per week | How to use the time |
|---|---:|---|
| Theory and mathematics | 3 | Understand concepts and derive important equations. |
| Coding and experiments | 4 | Implement concepts, change parameters, run experiments and debug. |
| Projects | 4 | Build independently and improve existing projects. |
| Review and documentation | 1 | Write notes, update your Git repository and explain what you learned. |
| **Total** | **12** | |

Use short exercises to test your knowledge, then use larger projects to find concepts you still do not understand. When you cannot explain what a piece of code does, investigate it instead of immediately moving to the next topic.

---

## 18. Final readiness checklist

Use this checklist to determine whether your learning is progressing beyond tutorials.

### Programming

- [ ] Write reusable Python code with tests and Git version control.
- [ ] Use SQL, HTTP APIs, virtual environments and debugging tools.

### Math and data

- [ ] Explain vectors, gradients, probability and cross-entropy.
- [ ] Clean datasets, prevent leakage and construct correct data splits.

### Machine learning

- [ ] Train and compare several classical ML models.
- [ ] Choose metrics, perform validation and explain model errors.

### Deep learning

- [ ] Implement forward propagation and understand backpropagation.
- [ ] Train an ANN using PyTorch and diagnose optimization problems.

### CNNs and Transformers

- [ ] Fine-tune a pretrained image model and evaluate it correctly.
- [ ] Explain attention, tokenization, masking and Transformer components.

### LLMs and RAG

- [ ] Use a pretrained LLM with structured outputs and evaluation.
- [ ] Build and evaluate a retrieval system with verifiable citations.
- [ ] Explain when fine-tuning is more appropriate than retrieval.

### Agents

- [ ] Build an agent with validated tools and explicit execution limits.
- [ ] Implement state, retries, tracing, human approval and security tests.

### Production

- [ ] Deploy an API with Docker and automated tests.
- [ ] Track model versions, evaluate performance and monitor costs.

---

## Final recommendation

Follow the curriculum through classical ML, learn PyTorch deeply, then specialize in LLM engineering, RAG and AI agents while continuing to develop software and deployment skills. That sequence gives you both the underlying knowledge and the practical skills needed to build increasingly capable AI systems.

The most important measure of progress is whether you can build something new without relying on a step-by-step tutorial. That is what turns knowledge of Python, ML, neural networks, CNNs, LLMs and agents into actual AI engineering skill.
