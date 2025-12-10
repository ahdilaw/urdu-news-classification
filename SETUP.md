# Quick Setup

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/urdu-news-classification.git
cd urdu-news-classification
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download FastText embeddings (optional, 7GB)
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.vec.gz
gunzip cc.ur.300.vec.gz
mv cc.ur.300.vec data/
```

## Running the Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
- `knn.ipynb` - K-Nearest Neighbors model
- `naive.ipynb` - Naive Bayes classifier
- `nn.ipynb` - Neural Network implementation

## Dataset

Due to size constraints, datasets are not included in the repository. You can:

1. **Use our scraped data**: Contact team members for access
2. **Scrape your own data**: Use scripts in `src/scraping/` (to be added)
3. **Download from paper sources**: See paper references

### Expected Data Structure

```
data/
├── raw-datasets/
│   ├── ary_data.csv
│   ├── dunya_data.csv
│   ├── express_data.csv
│   ├── geo_data.csv
│   └── jang_data.csv
├── processed/
│   ├── train.csv
│   ├── test_internal.csv
│   ├── test_dawn.csv
│   └── test_bbc.csv
└── eda/
    └── vocabulary.json  # Included in repo
```

## Quick Start Example

```python
import pandas as pd
from src.models import KNN

# Load data (after you have it)
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test_internal.csv')

# Train KNN
knn = KNN(k=4)
knn.fit(train_df['text'], train_df['label'])

# Evaluate
accuracy = knn.evaluate(test_df['text'], test_df['label'])
print(f"Accuracy: {accuracy:.2%}")
```

## Troubleshooting

**Issue**: FastText not working  
**Solution**: Ensure you have the Urdu vectors file (`cc.ur.300.vec`)

**Issue**: Out of memory  
**Solution**: Reduce batch size or use smaller subset of data

**Issue**: Missing data files  
**Solution**: Contact team or scrape fresh data using provided scripts
