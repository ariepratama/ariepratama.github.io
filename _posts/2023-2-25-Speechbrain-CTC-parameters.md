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


## Augmentation

### TimeDomainSpecAugment
is an augmentation that capable to: drop chunks of audio, or drop frequency band, or do speed perturbation. The default augmentation for my experiments is speed perturbation, where we speed up or speed down the input spectogram. Here is an example of where to use the augmentation.

```python
# wavs and wav_lens are loaded spectogram in form of pytorch
wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
# do augmentation directly to the wavs itself, 
# wav_lens is a must for this augmentation
# hparams is configuration loaded from speechbrain yaml
# hparams.augmentation will be initiated with TimeDomainSpecAugment
wavs = self.hparams.augmentation(wavs, wav_lens)
```

and here is the example of yaml, that will be translated into `hparams` object. 
```
augmentation: !new:speechbrain.lobes.augment.TimeDomainSpecAugment
    sample_rate: !ref <sample_rate>
    speeds: [95, 100, 105]
```

### SpecAugment
is an augmentation procedure that can do frequency masking, or time masking, or time warping. This augmentation will be applied to log mel spectogram, or latent feature. 

Frequency masking, is to 0 out the frequency, which is the `x` axis in log mel spectogram. While time masking, is to 0 out values along time axis, which is the `y` axis in log mel spectogram. Finally, time warping means to "remove" some sequential portion of the spectogram, or in other hand: "removing the values from time x1 to x2".

Here is the sample code, that describes where to put the augmentation with `wav2vec` model.

```python
wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
# Create latent feature from raw wavs with wav2vec2 model
feats = self.modules.wav2vec2(wavs)
# apply SpecAugment augmentation 
feats = self.hparams.augmentation(feats)
```

and here is the example on how to define the `SpecAugment` in speechbrain yaml file.
```
augmentation: !new:speechbrain.lobes.augment.SpecAugment
    freq_mask: True
    time_warp: True
    time_mask: True
```