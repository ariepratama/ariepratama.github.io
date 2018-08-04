---
layout: post
title: Virtebi Algorithm and Hidden Markov Model - Part 2
---


I've implemented virtebi algorithm and explain the advantage from naive approach at last post. Now it's time to look at another use case example: the Part of Speech Tagging! 

# POS Tag
Part of Speech Tag ([POS Tag](https://en.wikipedia.org/wiki/Part-of-speech_tagging) / Grammatical Tag) is a part of natural language processing task. The main problem is 
> "given a sequence of word, what are the postags for these words?". 


### Example of POS Tag 
If you understand this writing, I'm pretty sure you have heard categorization of words, like: noun, verb, adjective, etc. Well, those are POS Tag! But then we now have more POS Tag then you have teached in English! (you can see [here](http://universaldependencies.org/u/pos/))

For example this word:

    **"I like sushi"**

could be broken down into:

| I |like | sushi|
|--|--|--|
| PRON | VERB | NOUN |

Where,

PRON = pronoun

VERB = verb

NOUN = noun



# So how to answer the real question here? 

One way to model on how to get the answer, is by:
$$
P(pos\space tag \space sequence \space | \space words \space sequence) = \prod_{i=1}^{I}{P(tag\space |\space word_{w})}
$$


# Hidden Markov Model using Pomegranate
We can impelement this model with Hidden Markov Model. For this experiment, I will use [pomegranate](https://pomegranate.readthedocs.io/en/latest/HiddenMarkovModel.html#hiddenmarkovmodel) library instead of developing on our own code like on the post before. It will enable us to construct the model faster and with more intuitive definition. I will walk you through the process.



```python
import pandas as pd
# I will use nltk library with brown corpus as example
import nltk
import nltk.corpus as corp

import pomegranate as pm
```


```python
# download the corpus from nltk repository
if not corp.brown:
    nltk.download('brown')
```


```python
print('Brown corpus has {:,} tagged words / unigram'.format(len(corp.brown.tagged_words())))
```

    Brown corpus has 1,161,192 tagged words / unigram



```python
# see sneak peak from tagged words
corp.brown.tagged_words()[:10]
```




    [('The', 'AT'),
     ('Fulton', 'NP-TL'),
     ('County', 'NN-TL'),
     ('Grand', 'JJ-TL'),
     ('Jury', 'NN-TL'),
     ('said', 'VBD'),
     ('Friday', 'NR'),
     ('an', 'AT'),
     ('investigation', 'NN'),
     ('of', 'IN')]




```python
# sampling for the first 5000 tokens just for demo sake
sample = corp.brown.tagged_words()[:5000]
print('We have sample: {} tokens'.format(len(sample)))
```

    We have sample: 5000 tokens



```python
def construct_discrete_distributions_per_tag(tagged_tokens):
    """
    input:
    -------
    tagged_tokens: list of tagged tokens in form like:
            [('The', 'AT'),
             ('Fulton', 'NP-TL'),
             ('County', 'NN-TL'),
             ('Grand', 'JJ-TL'),
             ('Jury', 'NN-TL'),
             ('said', 'VBD'),
             ('Friday', 'NR'),
             ('an', 'AT'),
             ('investigation', 'NN'),
             ('of', 'IN')]
    
    This function will generate initial probability for each tag
    """
    tag_probs = dict()
    for token, tag in tagged_tokens:
        if tag not in tag_probs:
            tag_probs[tag] = dict()
            tag_probs[tag]['count_tag'] = dict()
            tag_probs[tag]['occurence'] = 1
        else:
            tag_probs[tag]['occurence'] += 1
            
        if token not in tag_probs[tag]['count_tag']:
            tag_probs[tag]['count_tag'][token] = 1
        else:
            tag_probs[tag]['count_tag'][token] += 1
            
    for tag in tag_probs:
        tag_probs[tag]['probs'] = dict()
        for token in tag_probs[tag]['count_tag']:
            tag_probs[tag]['probs'][token] = float(tag_probs[tag]['count_tag'][token]) / float(tag_probs[tag]['occurence'])
    
    return tag_probs            
```


```python
def construct_transition_probabilities_per_tag(tagged_tokens):
    """
    input:
    -------
    tagged_tokens: list of tagged tokens in form like:
            [('The', 'AT'),
             ('Fulton', 'NP-TL'),
             ('County', 'NN-TL'),
             ('Grand', 'JJ-TL'),
             ('Jury', 'NN-TL'),
             ('said', 'VBD'),
             ('Friday', 'NR'),
             ('an', 'AT'),
             ('investigation', 'NN'),
             ('of', 'IN')]
             
    This function will generate the emission matrix / probabilities for each tag
        
    """
    transition_probs = dict()
    for i in range(len(tagged_tokens)):
        current_tag = tagged_tokens[i][1]
        if current_tag not in transition_probs:
            transition_probs[current_tag] = dict()
            transition_probs[current_tag]['occurence'] = 0
            transition_probs[current_tag]['count_transition'] = dict()
        
        # evaluate previous tag
        if i > 0:
            previous_tag = tagged_tokens[i-1][1]
            pt = tagged_tokens[i-1][0]
            transition_probs[previous_tag]['occurence'] += 1
            
            # special case for <start> tag
            if pt == '.':
                if '<start>' not in transition_probs:
                    transition_probs['<start>'] = dict()
                    transition_probs['<start>']['occurence'] = 0
                    transition_probs['<start>']['count_transition'] = dict()
                if current_tag not in transition_probs['<start>']['count_transition']:
                    transition_probs['<start>']['count_transition'][current_tag] = 0
                    
                transition_probs['<start>']['count_transition'][current_tag] += 1
                transition_probs['<start>']['occurence'] += 1
                    
            
            #init
            if current_tag not in transition_probs[previous_tag]['count_transition']:
                transition_probs[previous_tag]['count_transition'][current_tag] = 0
                
            transition_probs[previous_tag]['count_transition'][current_tag] += 1
    for tag in transition_probs:
        transition_probs[tag]['probs'] = dict()
        for transit_tag in transition_probs[tag]['count_transition']:
            transition_probs[tag]['probs'][transit_tag] = \
                float(transition_probs[tag]['count_transition'][transit_tag]) / float(transition_probs[tag]['occurence'])
    
    return transition_probs
```


```python
def build_hmm_model(token_dist, transition_dist, model_name='hmm-tagger'):
    state_dict = dict()
    for token in token_dist:
        state_dict[token] = \
            pm.State(
                pm.DiscreteDistribution(
                    token_dist[token]['probs']
                )
                , name=token
            )
            
    model = pm.HiddenMarkovModel(model_name)
    model.add_states(list(state_dict.values()))
    
    # initialization for starting tokens
    for token, prob in transition_dist['.']['probs'].items():
        model.add_transition(state_dict[token], model.end, prob)
        
    for token, prob in transition_dist['<start>']['probs'].items():
        model.add_transition(model.start, state_dict[token], prob)
    
    transition_dist_list = list(transition_dist.items())
    for i in range(1, len(transition_dist_list)):
        ptoken, pmeta = transition_dist_list[i]
        if ptoken != '.' and ptoken != '<start>':
            for ctoken, cprob in pmeta['probs'].items():
                
                model.add_transition(
                    state_dict[ptoken],
                    state_dict[ctoken],
                    cprob
                )

        
    return model, state_dict
```


```python
from sklearn.base import BaseEstimator, ClassifierMixin 
```


```python
class HmmTaggerModel(BaseEstimator, ClassifierMixin):
    """
    POS Tagger with Hmm Model
    """
    def __init__(self):
        self._inner_model = None
        self._tag_dist = None
        self._transition_dist = None
        self._state_dict = None
    
    def fit(self, X, y=None):
        """
        expecting X as list of tokens, while y is list of POS tag
        """
        combined = list(zip(X, y))
        self._tag_dist = construct_discrete_distributions_per_tag(combined)
        self._transition_dist = construct_transition_probabilities_per_tag(combined)
        
        self._inner_model, _ = build_hmm_model(self._tag_dist, self._transition_dist)
        self._inner_model.bake()
        
    
    def predict(self, X, y=None):
        """
        expecting X as list of tokens
        """
        return [state.name for i, state in self._inner_model.viterbi(X)[1]][1:-1]
```


```python
def accuracy(ypred, ytrue):
    total = len(ytrue)
    correct = 0
    for pred, true in zip(ypred, ytrue):
        if ypred == ytrue:
            correct += 1
    
    return correct / total
```


```python
model = HmmTaggerModel()
model.fit(map(lambda x: x[0], sample), map(lambda x: x[1], sample))
```


```python
import numpy as np
```


```python
st_idx = [0, 68]
end_idx = [10, 72]

list_acc = list()
for st, end in zip(st_idx, end_idx):
    actual_words = list(map(lambda x: x[0], sample[st:end]))
    actual = list(map(lambda x: x[1], sample[st:end]))
    predicted = model.predict(
            list(map(lambda x: x[0], sample[st:end]))
        )
    print('actual words: {}'.format(' '.join(actual_words)))
    print('actual tags: {}'.format(' '.join(actual)))
    print('predicted tags: {}'.format(' '.join(predicted)))
    print('============================================================')
    list_acc.append(accuracy(actual, predicted))
mean_acc = np.mean(list_acc)

print('mean accuracy: {}'.format(mean_acc))
```

    actual words: The Fulton County Grand Jury said Friday an investigation of
    actual tags: AT NP-TL NN-TL JJ-TL NN-TL VBD NR AT NN IN
    predicted tags: AT NP-TL NN-TL JJ-TL NN-TL VBD NR AT NN IN
    ============================================================
    actual words: The September-October term jury
    actual tags: AT NP NN NN
    predicted tags: AT NP NN NN
    ============================================================
    mean accuracy: 1.0


That went well! Now lets try for bigger corpuses! I wil use 500,000 words from the brown corpus.


```python
larger_sample = corp.brown.tagged_words()[:500000]
model = HmmTaggerModel()
model.fit(
    map(lambda x: x[0], larger_sample), 
    map(lambda x: x[1], larger_sample)
)
```


```python
st_idx = [0, 68, 103]
end_idx = [10, 72, 118]

list_acc = list()
for st, end in zip(st_idx, end_idx):
    actual_words = list(map(lambda x: x[0], sample[st:end]))
    actual = list(map(lambda x: x[1], sample[st:end]))
    predicted = model.predict(
            list(map(lambda x: x[0], sample[st:end]))
        )
    print('actual words: {}'.format(' '.join(actual_words)))
    print('actual tags: {}'.format(' '.join(actual)))
    print('predicted tags: {}'.format(' '.join(predicted)))
    print('============================================================')
    list_acc.append(accuracy(actual, predicted))
mean_acc = np.mean(list_acc)

print('mean accuracy: {}'.format(mean_acc))
```

    actual words: The Fulton County Grand Jury said Friday an investigation of
    actual tags: AT NP-TL NN-TL JJ-TL NN-TL VBD NR AT NN IN
    predicted tags: AT NP-TL NN-TL JJ-TL NN-TL VBD NR AT NN IN
    ============================================================
    actual words: The September-October term jury
    actual tags: AT NP NN NN
    predicted tags: AT NP NN NN
    ============================================================
    actual words: `` Only a relative handful of such reports was received '' , the jury said
    actual tags: `` RB AT JJ NN IN JJ NNS BEDZ VBN '' , AT NN VBD
    predicted tags: `` RB AT JJ NN IN JJ NNS BEDZ VBN '' , AT NN VBD
    ============================================================
    mean accuracy: 1.0


It does make a good model!! Though it takes more time for larger model. 

## Is This a Perfect Model?
But now you might wonder, is this the perfect model for POS tagging? Well congratulation! you asked a good question and the answer is No! It is not the perfect model. I have only showed you working case, but actually there will be problems:
1. Out Of Vocabulary (OOV), where there are no matching word between input and training data. 
2. Training data without exhaustive positioning of tagging. This simple model will not be able to adjust it self if, let's say there is a word tagged as NP in the beginning of sentence, but this has never happened in training data. 
3. Lack of preprocessing, such as lower casing, punctuation removal, etc will make the model not focused enough into predicting POS tag.

# Conclussion
Even though HMM will produces a fairly good model for POS Tagging, but you need to watch for disadvantages for using this model.

# References:
- https://en.wikipedia.org/wiki/Brown_Corpus
- https://pomegranate.readthedocs.io/en/latest/HiddenMarkovModel.html#hiddenmarkovmodel

