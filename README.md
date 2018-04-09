# PairCNN-Ranking
A pytorch implementation of [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

This version is implemented with reference of [zhangzibin](https://github.com/zhangzibin/PairCNN-Ranking)

## Training Data
As **train.txt** and **test.txt** in **./data** dir, data sample is separated by line separator,
and the sample itself is splited by comma: query, document, label.
And the example data is created by me to test the code, which is not real click data.

## Usage
```bash
python main.py help
```

## Project Structure

The file holder structure of this project is:

```
├── checkpoints/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── initfunctional.py
│   ├── BasicModule.py
│   └── PairCNN.py
├── utils/
│   ├── __init__.py
│   └── visualize.py
├── __init__.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
```
