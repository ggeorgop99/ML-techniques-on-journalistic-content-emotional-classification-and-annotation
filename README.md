# Greek Sentiment Analysis for Journalistic Content

This repository contains the code and resources for a diploma thesis project at the Department of Electrical and Computer Engineering, Faculty of Engineering, Aristotle University of Thessaloniki. The project focuses on sentiment analysis and text summarization of Greek text using various machine learning and algorithmic techniques.

## Project Overview

The project consists of three main components:

1. **Dataset Creation**: Building a large sentiment-annotated dataset in Greek by scraping reviews from the online shopping website "Skroutz"
2. **Sentiment Analysis**: Implementing and comparing various algorithmic techniques (both ML and non-ML) for analyzing sentiment in Greek text
3. **Text Summarization**: Implementing multiple approaches for summarizing Greek text, including:
   - TextRank-based extractive summarization
   - BART-based abstractive summarization
   - GPT-based abstractive summarization
   - Hierarchical summarization using Greek T5 models

## Repository Structure

```
├── mysenti.py                 # Python implementation of SentiStrength algorithm
├── reviews.csv               # 2800 reviews dataset
├── stars.csv                 # Star ratings for each review
├── finallexformysenti/      # Lexicon files for sentiment analysis
├── dataset/                 # Dataset files
│   ├── dirtyreviews.csv     # Raw crawled data
│   ├── reviewstars.csv      # Cleaned review data
│   └── results.csv          # Analysis results
├── skroutz_scraping/        # Web scraping components
│   └── spiders/
│       ├── amazon_reviews.py
│       └── amazon_reviews22.py
├── neuralnet/               # Machine learning models and results
│   ├── training scripts
│   ├── trained models
│   └── results and graphs
└── summarization/          # Text summarization components
    ├── ranking_summarization.py    # TextRank implementation
    ├── bart_summarization.py       # BART model implementation
    ├── gtp_summarization.py        # GPT-based summarization
    ├── new_bart_sum.py            # Hierarchical summarization
    └── summaries_hierarchical/    # Generated summaries
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation.git
cd ML-techniques-on-journalistic-content-emotional-classification-and-annotation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Scraping

To collect data from Skroutz, run the following commands from the `amazon_reviews_scraping` directory:

```bash
scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews.py -o links.csv
scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews22.py -o dirtyreviews.csv
```

### Data Processing

The `mysenti.py` script contains two important functions:
- `clearfiles()`: Processes raw data to create `reviewstars.csv` suitable for SentiStrength analysis
- `splitfiles()`: Splits the data into `reviews.csv` and `stars.csv`

Note: `clearfiles()` requires `dirtyreviews.csv` to be present in the directory.

### Sentiment Analysis

The neural network models and analysis scripts are located in the `neuralnet` directory. This includes:
- Training scripts for various ML models
- Pre-trained models
- Analysis results and visualizations

### Text Summarization

The project includes multiple approaches for summarizing Greek text:

1. **TextRank Summarization** (`ranking_summarization.py`):
   - Extractive summarization using the TextRank algorithm
   - Suitable for shorter texts and quick summaries

2. **BART Summarization** (`bart_summarization.py`):
   - Abstractive summarization using the BART model
   - Good for medium-length texts

3. **GPT-based Summarization** (`gtp_summarization.py`):
   - Abstractive summarization using GPT models
   - Requires OpenAI API key
   - Best for high-quality, creative summaries

4. **Hierarchical Summarization** (`new_bart_sum.py`):
   - Two-stage summarization process using Greek T5 models
   - Handles long texts by breaking them into chunks
   - Supports batch processing for efficiency
   - Saves summaries with metadata in the `summaries_hierarchical` directory

Example usage of hierarchical summarization:
```python
from summarization.new_bart_sum import summarize_tweets_transformer_greek

# Load and preprocess data
df = load_and_preprocess_greek("input.csv", text_column="text")

# Generate summary
summary = summarize_tweets_transformer_greek(
    df,
    model_name="IMISLab/GreekT5-mt5-small-greeksum",
    max_chunk_length_words=400,
    summary_max_length=141,
    summary_min_length=40
)
```

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Verus Plus - Department of Electrical and Computer Engineering, Aristotle University of Thessaloniki

## Acknowledgments

- SentiStrength algorithm implementation based on [SentiStrength](http://sentistrength.wlv.ac.uk/)
- Greek T5 models from [IMISLab](https://huggingface.co/IMISLab)
- TextRank implementation based on the [sumy](https://github.com/miso-belica/sumy) library
- BART and GPT implementations using [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- Scraping framework using [Scrapy](https://scrapy.org/)
