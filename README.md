# Classification of Urdu News Articles

> Machine Learning project for classifying Urdu news articles into five categories using KNN, Naive Bayes, and Neural Networks

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-completed-success)

Research paper for CS 438 Machine Learning course at LUMS, exploring challenges and insights in low-resource language classification.

---

## Abstract

This study addresses the classification of Urdu news articles into five categories: **Entertainment**, **Business**, **Sports**, **Science-Technology**, and **International**. 

Leveraging a dataset scraped from seven major Urdu news websites (5,841 articles), we employed a robust methodology encompassing:
- Data preprocessing and vocabulary curation
- Feature transformation (TF-IDF, Count Vectorizer, FastText embeddings)
- Three machine learning models: K-Nearest Neighbors (KNN), Naive Bayes, and custom Neural Network

**Key Findings:**
- KNN achieved best overall performance (81.62% on Dawn dataset)
- All models struggled with distribution shift (BBC dataset: ~62% accuracy)
- FastText embeddings showed only 2% improvement, highlighting low-resource language challenges

---

## Team

- **Ahmed Wali** - 26100264@lums.edu.pk
- **Muhammad Ismail Humayun** - 25020267@lums.edu.pk
- **Hamza Sherjeel** - 26100361@lums.edu.pk
- **Yamsheen Saqib** - 26100379@lums.edu.pk
- **Shanzay Omer** - 26100202@lums.edu.pk

---

## Dataset

### Training Data (5,943 articles)
- **Sources**: ARY, Dunya News, Express, Geo, Jang
- **Size**: 18.75 MB
- **Distribution**:
  - Sports: 1,387
  - Science-Technology: 1,320
  - International: 1,227
  - Entertainment: 1,123
  - Business: 886

### External Test Sets
- **Dawn Dataset** (265 articles, 1.34 MB) - Similar distribution to training
- **BBC Dataset** (1,177 articles, 12.14 MB) - Longer articles, different distribution

All datasets available in `data/` directory.

---

## Methodology

### 1. Data Collection
- Scraped from 7 major Urdu news websites
- Custom scripts with randomized delays for sources like ARY
- Multiple categories to ensure balanced representation

### 2. Data Cleaning
- Removed non-Urdu words, punctuation, and extra spaces
- Vocabulary: 51,074 unique Urdu words (after stopword removal)
- Stopwords: 517 from Kaggle + manual annotation
- **Cohen's Kappa: 0.664** (substantial inter-annotator agreement)

### 3. Feature Extraction
- **TF-IDF Vectorizer**: Unigrams, bigrams, trigrams
- **Count Vectorizer**: Word occurrence encoding
- **FastText Embeddings**: 300-dimensional semantic vectors

### 4. Models Implemented

#### K-Nearest Neighbors (KNN)
- TF-IDF features with cosine similarity
- Optimal k=4 (combined dataset) / k=6 (internal dataset)
- **Best Performance**: 93% internal, 82% Dawn, 62% BBC

#### Naive Bayes
- Unigram features
- Custom implementation from scratch
- **Performance**: 93% internal, 82% Dawn, 61% BBC

#### Neural Network
- Custom NumPy implementation
- Architecture: 256 → 128 → output layer
- Batch size: 32, Epochs: 250
- **Performance**: 96% training, 87% internal, 68% combined-BBC

---

## Results

### Model Comparison

| Model | Internal Test | Dawn Test | BBC Test |
|-------|--------------|-----------|----------|
| **KNN** | 93% | 82% | 62% |
| **Naive Bayes** | 93% | 82% | 61% |
| **Neural Network** | 87% | 71% | 53% |
| **NN (Combined)** | 82% | 88% | 68% |
| **NN (Embeddings)** | 90% | - | 57% |

### Key Insights
- Models perform well on similar distributions (Dawn)
- Significant accuracy drop on different distributions (BBC)
- "International" category shows lowest precision/recall (label overlap)
- FastText embeddings provide minimal improvement (2%)

---

## Visualizations

### PCA Analysis
Shows visible feature clusters for word grouping.

### t-SNE Analysis
Clear clusters indicating nearest neighbor methods effectiveness.

### Confusion Matrix
Available in results showing per-category performance.

---

## Project Structure

```
.
├── data/
│   ├── raw-datasets/          # Original scraped data
│   ├── eda/                   # Exploratory data analysis
│   │   ├── vocabulary.json    # 51,074 unique words
│   │   └── stopwords/         # Annotated stopwords
│   └── processed/             # Clean datasets
├── models/
│   ├── knn/                   # K-Nearest Neighbors
│   ├── naive_bayes/           # Naive Bayes classifier
│   └── neural_network/        # Custom NN implementation
├── notebooks/
│   ├── data_collection.ipynb
│   ├── eda.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   └── models/
├── results/
│   ├── confusion_matrices/
│   └── performance_metrics/
└── paper/
    └── Team9_Report.pdf       # Full research paper
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/urdu-news-classification.git
cd urdu-news-classification

# Install dependencies
pip install -r requirements.txt

# Download FastText Urdu embeddings (optional)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.vec.gz
```

---

## Usage

### Data Preprocessing
```python
from src.preprocessing import clean_data, build_vocabulary

# Clean the dataset
clean_data('data/raw-datasets/combined.csv', 'data/processed/clean.csv')

# Build vocabulary
vocab = build_vocabulary('data/processed/clean.csv')
```

### Train Models
```python
from src.models import KNN, NaiveBayes, NeuralNetwork

# Train KNN
knn = KNN(k=4)
knn.fit(X_train, y_train)
accuracy = knn.evaluate(X_test, y_test)

# Train Naive Bayes
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Train Neural Network
nn = NeuralNetwork(hidden_layers=[256, 128])
nn.train(X_train, y_train, epochs=250, batch_size=32)
```

---

## Key Challenges

### Low-Resource Language
- Limited pre-trained embeddings for Urdu
- FastText embeddings showed minimal improvement (2%)
- Need for Urdu-specific language models

### Distribution Shift
- Models trained on specific sources struggle with new distributions
- BBC dataset (longer articles) caused significant accuracy drop
- Domain adaptation remains a challenge

### Label Overlap
- "International" category frequently mislabeled
- Overlap with other categories in training data

---

## Future Work

1. **Advanced Architectures**: Explore transformers and BERT-like models for Urdu
2. **Transfer Learning**: Use pre-trained Urdu language models
3. **Dataset Diversity**: Increase variety in training data sources
4. **Label Refinement**: Address ambiguities in "International" category
5. **Urdu Embeddings**: Develop better embedding models for Urdu
6. **Domain Adaptation**: Techniques to handle distribution shift

---

## Related Projects

**ChatGPT RAG Injector**: [github.com/ahdilaw/chatgpt-rag-injector](https://github.com/ahdilaw/chatgpt-rag-injector)
- Built during Summer 2024 Research Internship
- Transparent context injection into ChatGPT using mitmproxy
- Could be used to inject Urdu context for better LLM performance

---

## References

1. **Urdu Stopwords List** - T. Atman (2019)  
   [Kaggle Dataset](https://www.kaggle.com/datasets/rtatman/urdu-stopwords-list)

2. **FastText Crawl Vectors** - Bojanowski et al. (2017)  
   [FastText Urdu Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@inproceedings{wali2024urdu,
  title={Classification of Urdu News Articles: Challenges and Insights in a Low-Resource Language},
  author={Wali, Ahmed and Humayun, Muhammad Ismail and Sherjeel, Hamza and Saqib, Yamsheen and Omer, Shanzay},
  booktitle={Proceedings of Fall 2024 (CS 438 Machine Learning Group Project)},
  year={2024},
  organization={LUMS}
}
```

---

## Acknowledgments

Special thanks to:
- Course instructor and TAs at LUMS
- Kaggle community for Urdu stopwords dataset
- FastText team for pre-trained embeddings
- All team members for their contributions

---

**Institution**: Lahore University of Management Sciences (LUMS)  
**Course**: CS 438 Machine Learning  
**Semester**: Fall 2024
