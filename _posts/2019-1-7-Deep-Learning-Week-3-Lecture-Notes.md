---
layout: post
title: Deep Learning - Week 3 Lecture Notes
tags:
    - coursera
    - deep-learning
    - lecture-notes
---

# Basic Architecture of Neural Network
<div class="mermaid">
graph LR
  subgraph input layer
    X1[x1]
    X2[x2]
    X3[x3]  
  end
  subgraph hidden layer
    H1((h1))
    H2((h2))
    H3((h3))
  end
  subgraph output layer
    YHAT((yhat))
  end 
  X1 --> H1
  X1 --> H2
  X1 --> H3
  X2 --> H1
  X2 --> H2
  X2 --> H3
  X3 --> H1
  X3 --> H2
  X3 --> H3
  H1 --> YHAT((yhat))
  H2 --> YHAT
  H3 --> YHAT
  
</div>

This example ilustrate **2 Layer Neural Network** because we do not count input layer. the hidden layers can be think as multiple logistic regression nodes that passing output to one another. 

Using superscript like $^{[1]}$ denotes which layer will be pointed, for example in the picture above, input layer is $^{[1]}$, hidden layer is $^{[2]}$, and output layer is $^{[3]}$.

||||
| --- | --- | --- |
|$z^{[1]}$|=| $W^{[1]}x + b^{[1]} $|
|$a^{[1]}$|=| $ \sigma ( z^{[1]} ) $|
|$z^{[2]}$|=| $ W^{[2]}a^{[1]} + b^{[2]} $|
|$a^{[2]}$|=| $ \sigma ( z^{[2]} ) $|


but all these operations must be repeated by $n$ training sample, so in order to do that faster, we need to vectorize these operations.


# Activation Functions

## Sigmoid

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Andrew's has rarely use sigmoid activation function for hidden units, he prefer tanh for these hidden units. However, sigmoid function might be used in output layer if the ouput is binary. 

## Tanh

$$
a = tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

Centering the mean towards zero will make hidden units faster to converge.


## ReLU

Rectified Linear Units

$$
a = ReLU(z) = max(0, z)
$$


if you don't know what activation function to use, **use this**.

### Leaky ReLU

$$
a = LReLU(z) = max(0.01 z, z)
$$

But why $0.01$, sometimes just work! (no idea or whatsoever)


## Why Neural Network Need Non-Linear Function?

||||
| --- | --- | --- |
|$z^{[1]}$|=| $W^{[1]}x + b^{[1]} $|
|$z^{[2]}$|=| $ W^{[2]}z^{[1]} + b^{[2]} $|
|$z^{[2]}$|=| $ W^{[2]} (W^{[1]} x^{[1]} + b^{[1]}) + b^{[2]} $|
|$z^{[2]}$|=| $ (W^{[2]} W^{[1]}) x + (b^{[1]} + b^{[2]}) $|
|$z^{[2]}$|=| $ W^{'} x + b^{'} $|

Turns out we only computing linear function! No matter how many layers that we've put in, duh!.
![gif](https://media.giphy.com/media/nlIZ0vL7AjaMIXHZjj/giphy.gif)



