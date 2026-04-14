# X Large Language Model

This repository contains the code for developing, pretraining and finetuning a LLM.

## Objective

The goal of the XLANG project is to build a LLM from basic principles.
Its implementation relies on the PyTorch deep learning library.

The purpose of a LLM is to understand, generate and respond to human-like text.

## Acknowledge

The experiments were based on content from Sebastien Raschka's book 
"Build a Large Language Model from Scratch", Manning Press.

## Process

1 - Pretraining: The LLM is initially trained by scanning a large amout of text to form the base model.
2 - Fine-tuning: The pretrained model is further trained on labeled data.

## Architecture

A transformer architecture is used, consisting of an encoder, decoder and a self-attention mechanism.
The encoder transforms input text into numeric vectors.
The decoder generates output text from the numeric vectors.
The self-attention mechanism allows the model to weigh the importance of different words 
in a sequence relative to each other.

Although it's possible to use pretrained models such as Word2Vec to generate embeddings, the
alternative of using a custom embedding allows the model to be optimized to a specific task and data.

