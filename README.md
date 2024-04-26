# NLP-Recommender
Recommendation system reliant on a user's other reviews in predicting their rating for a product.

## Contributors
* Brian Tsai - tsai209@purdue.edu
* Ryan Shue - rshue@purdue.edu
* Laura Zhou - lczhou@purdue.edu

## Dataset
`Video_Games_5.json` can be downloaded from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ , or via this direct url: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz

Once downloaded, this file should be placed in the `Preprocessing` folder. Do not unzip the file.

## FastText Embeddings

FastText embeddings should be downloaded from this url: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

Once downloaded, this file should be placed next to the `ML_LSTM.ipynb` notebook.

## Preprocessing
The data is already preprocessed and located at `Preprocessing/dataset_operating.csv` (if not, you can download from the Unavailable Data link the appendix). To remake this file, change your directory into `Preprocessing` and execute `preprocess.py`. Remember that the `Video_Games_5.json.gz` needs to be in that directory, still zipped! The preprocessing removes duplicate or empty reviews, and requires every user to have at least six reviews.

## Environment
The environment assumes you installed PyTorch 2.2.0 or above compiled with the necessary CUDA installation. When using CUDA, ensure you have at least 8GB of memory in your GPU. If you want to use the CPU instead, replace existences of `torch_device = torch.device("cuda")` with `torch_device = torch.device("cpu")`.

In addition, install the following packages:

* `jupyter-notebook` or `jpyterlab` to run the code.
* `pandas` for dataframe manipulation
* `prettytable` for the part where the number of parameters is displayed
* `scikit-learn` for the `train_test_split` and product embedding visualization
* `tqdm` for progress bars.
* `torchmetrics` for R2 metrics
* `transformers` Optional, for the transformer code (see appendix)

## Code Execution

The relevant code to the project is located at `ML_LSTM.ipynb`.

# Appendix

## Unavailable data
If data is unavailable, you can tryi this shared onedrive link: https://purdue0-my.sharepoint.com/:f:/g/personal/tsai209_purdue_edu/Ens8h2sboOpCqxic5Mf0lvwB7ExrT5tq90c_pfgzscqi0Q?e=y3bgER

## Transformers

The transformer network is located at `ML_Transformers.ipynb`. This was an earlier iteration where a Bert Transformer instead of an LSTM was used instead to process the text data in reviews. Unfortunately, due to the complete model's 113 million parameters, limited compute and memory requirements prevented its full use. The program was tested and confirms to work a on Google Colab NVIDIA L4 Tensor core GPU, and its 24GB of memory was unable to handle batch sizes greater than four. Each epoch took around 90 minutes to complete.