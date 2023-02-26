---
layout: post
title: Speechbrain CTC parameters
tags:
    - nlp
    - python
    - speechbrain
    - speech
---
Speechbrain is a framework to experiments with neural network in speech. It has components and structure that is usually used for speech tasks, and nicely integrated with pytorch. They have lots of recipes / quickstart configurations for certain speech tasks, but their documentation is not there yet. In this article I'll explain my exploration for speechbrain parameters, especially for automatic speech recognition (ASR) with connectiionist temporal classification (CTC) loss. 

## Tokenizer parameters
Speechbrain uses `SentencePiece` tokenizer, and in the yaml example, it has 2 configurations related to this tokenizer `token_type` and `character_coverage`.

`token_type` is `Sentenpiece.model_type` which currently has 3 mode_type: `char`, `unigram`, `bpe`. While `char` and `unigram` is self-explainatory, `bpe` will download Google's unsupervised sentencepiece and that will be used to tokenize the target text. 


`character_coverage` is to be used in conjunction with `token_type=bpe`, which will indicates the percentage or fraction of characters that the sentencepiece model should cover. `1.0` means 100% characters will be covered, and this is the default value, since it does make sense to cover all the characters in a language. However, when using the SentencePiece tokenizer for rich languages, like Chinese or Japanese [the documentation](https://github.com/google/sentencepiece) itself mention to reduce this towards `0.9995`.