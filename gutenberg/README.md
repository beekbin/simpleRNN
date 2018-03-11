# Introduction
[Project Gutenberg](https://www.gutenberg.org) offers over 56,000 free eBook (by March 2018), 
and each book has **Plain Text UTF-8** version. So it is a ideal source to play for NLP problems.

Here this python script is used to generate word sequences of sentences from the books. 
The sequences will be fed to RNN, and RNN will try to learn to generate new words.


# Usage

#### Step1 Download a **Plain Text UTF-8** version book.
For example, download the [__Alice's Adventures in Wonderland__](http://www.gutenberg.org/cache/epub/19033/pg19033.txt)

#### Step2 Generate Vocabulary and training data
```termnial
cd gutenberg && mkdir data
python main.py ../data/alice.txt
```
