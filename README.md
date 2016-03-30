# PrimeLM 

PrimeLM is a flexible and reusable feed-forward neural network which can be used to train neural language models and joint models (Devlin et. al, 2014), and interface with popular SMT systems like Moses (http://www.statmt.org/moses/). It is implemented in Python using Theano, which makes is easy-to-use and modify. 

## Features

* Implementation of self-normalized log-likelihood (Devlin et. al, 2014)  and noise contrastive estimation (NCE) loss functions, to train fast neural language models.
* Decoder Integration with MOSES using NeuralLM and BilingualLM feature functions in MOSES. Also, rescoring MOSES n-best lists using neural language models. 
* Efficient and optimized implementation using Theano, capable of using GPU support for faster training and decoding. 
* The neural network architecture is flexible. Multiple hidden layers and various activation function, multiple sets of features with different embeddings etc.
* The training is also flexible, with layer specific and adjustable learning rates, using various cost functions like log-likelihood and NCE and regularizations (L1 and L2). 
* Support scripts to rescore n-best lists from MOSES, convert models to and from NPLM format and oracle score computation. 
* Preprocessing scripts for monolingual language modeling and bilingual language modeling. 

## Getting Started






