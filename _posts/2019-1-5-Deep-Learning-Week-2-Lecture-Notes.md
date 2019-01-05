---
layout: post
title: Deep Learning - Week 2 Lecture Notes
tags:
    - coursera
    - deep-learning
    - lecture-notes
---
This is my notes for [Deep Learning Course in Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/home/week/2). I jumped straight to week 2 because week 1 is about introduction that I've known. Week 2 in summary is structured as: starting from binary classification with logistic regression, loss function and cost function, computational graph.

## Binary Classification
in image classification you will have to process and image that would most probably consists of RGB image. 

### Notation

$$
(x, y), x \in R^{n_{x}}
$$

$m$ training example denoted by 

$$
 (x^{(i)}, y^{(i)}) 
$$



### Logistic Regression
Given x, want $\hat y = P (y=1 \mid x)$

output $\hat y = \sigma(w^{T}x  + b)$


sigmoid function with input variable z definition:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

### Logistic Regression Cost Function
Usually we use loss function as follow

$$L(\hat y, y) = \frac{1}{2}(\hat y-y)^2$$

but this is non-convex function, so no single solution exists. We need to find another loss function for logistic regression!

$$L(\hat y, y) = -(y \log \hat y + (1-y) \log (1-\hat y))$$

Cost function

$$J(w,b) = \frac{1}{m} \sum_{i}^{m} L (\hat y^{i}, y^{i})$$


### Gradient Descent
for training or learning how to get logistic regression parameters.

```
Repeat
w := w - \alpha dJ(w) / dw
```
where $\alpha$ is the learning rate


## Computation Graph
Computation graph explains the use of backpropagation mechanism. Backward function computes the gradient of a cost function. Backpropagation distribute/propagate the gradient of a cost function to all the layers before.

assume we have function $J(a,b,c) = 3(a + bc)$, then we further define:

$u = bc$

$v = a + u$

$J = 3v$

<div class="mermaid">
graph LR
  B[b=3] --> U[u=bc -> 6]
  C[c=2] --> U
  A[a=5] --> V[v=a+u -> 11]
  U --> V
  V --> J[J=3v -> 33]

</div>

then we want to optimize function $J$, which in the past lectures, $J$ is the cost function. Remember the difference between cost function and loss function: cost funtion is the average of loss function, while loss function is the single instance loss or error made by our prediction.

Since we have this graph, now one of the question that we would like to answer is 
> how much of $J$ value change if we change $a$

from $J = 3v$ we know that:

$$
\frac{dJ}{dv} = 3
$$

now we wanted to answer $\frac{dJ}{da} = ?$

we know that $a$ affect $v$ and $v$ affect $J$ 
<div class="mermaid">
graph LR
  A[a] --> V[v]
  V --> J[J]
</div>

so following the chain rule:

$$
\frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da}
$$

we have $\frac{dJ}{dv} = 3$ and $\frac{dv}{da} = 1$ (this from calculus derivative, for more practical explanation please see the lecture video). Then we got:

$$
\frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da} = (3)(1) = 3
$$

What does this means? it means if we change the value of $a$ by some number, it would change the final function $J$ in magnitude of 3. Say if we change the value of $a$ to $5.0001$, then $v = 11.001$ and $J = 33.003$.


## Incorporating Computation Graph Into Logistic Regression
we have

$$
z = w^{T}x + b
$$

$$
\hat y = a = \sigma(z)
$$

$$
L(\hat y, y) = - (y \log(\hat y) + (1-y)\log(1-\hat y))
$$

then we can construct computation graph as follow

<div class="mermaid">
graph LR
  X1[x1] --> Z[z=w1 x1 + w2 x2 + b]
  W1[w1] --> Z
  X2[x2] --> Z
  W1[w2] --> Z
  B[b] --> Z
  Z --> Y[yhat = σ of z]
  Y --> L[L yhat, y]
</div>

so we got (like Andrew Ng said: don't worry about the calculus now)

$$
da = \frac{dL(a,y)}{da} = - \frac{y}{a} + \frac{1-y}{1-a}
$$

$$
dz = \frac{dL}{dz} = \frac{dL(a,y)}{da} \frac{da}{dz} = (- \frac{y}{a} + \frac{1-y}{1-a}) (a (1-a)) = a - y
$$

$$
dw_1 = \frac{dL}{dw_1} = x_1 \space dz
$$

$$
dw_2 = \frac{dL}{dw_2} = x_2 \space dz
$$

then we can update weights as follow:

$$
w_1 = w_1 - \alpha \space dw_1
$$

$$
w_2 = w_2 - \alpha \space dw_2
$$

where $\alpha$ is the learning rate that we set. Well this all is for **single training set**, if we wanted to calculate, say $w1$ from overall cost function, where:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(a^i, y^i)
$$

then

$$
\frac{d}{d1} J(w_1, b) = \frac{1}{m} \sum_{i=1}^{m} \frac{d}{w1} L(a^i, y^i)
$$