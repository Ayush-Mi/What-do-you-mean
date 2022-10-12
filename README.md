# What-do-you-mean
This is an implementation of intent classification problem in English sentences.

## Description
This repo implements and intent detection model using BERT embeddings. The dataset is taken from [SNIPS](https://github.com/jianguoz/Few-Shot-Intent-Detection) which has more than 13,000 data points with 7 intents in training set and 700 data points in each validation and test set. The data distribution per intent is almost same with each having more than 1800 sentences.

The model uses BERT-en with 12 headers to generate (None, 768) word embeddings and also uses the tensorflow hub API to preprocess the sentences.

After training I tried to compress the model using tensorflow quantization API which reduced the model size to a quarter of its original for Dynamic Range Quantization and to half for Float16 quantization. However, the inference using these compressed models threw error as the operations on the models from tf-hub were not able to execute in tf-lite format.

## Dependencies
The required libraries can be installed from requirements.txt file.

## Results
With the help of BERT embeddings, the model was able to coverge in just 5 epochs.

![](https://github.com/Ayush-Mi/What-do-you-mean/blob/main/train_results.png)

Performance on test data

| Model Size | Inference Time | Accuracy | F1-score | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 439.5 MB | 0.12 sec | 90.85 | 88.99 | 88.99 | 88.99 |
