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