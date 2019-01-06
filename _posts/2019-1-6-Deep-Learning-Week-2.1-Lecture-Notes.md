---
layout: post
title: Deep Learning - Week 2.1 Lecture Notes
tags:
    - coursera
    - deep-learning
    - lecture-notes
---
Short introduction to vector operation in python numpy in logistic regression.

# Vectorizing Logistic Regression
## Computing
Computing logistic regression
```python
np.dot(w.T, x) + b
```
where `b` is a single real number or a `float` in python, that will be *broadcasted* to all element in matrix.

## Vectorizing Backpropagation

```python
# b will be broadcasted
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
dz = A - Y
# m will be broadcasted
dw = np.dot(X, dz.T) / m
db = np.sum(dz) / m

# update weights
w = w - alpha * dw
b = b - alpha * db

```

# Broadcasting in Python (numpy)

Given this table, where the values are calories
||Apples|Beef|Eggs|Potatoes|
|---|---:|---:|---:|---:|
|Carb|56.0|0.0|4.4|68.0|
|Protein|1.2|104.0|52.0|8.0|
|Fat|1.8|135.0|99.0|0.9|

> Calculate % of calories from Carb, Protein, Fat without for loops

```python
cal = A.sum(axis=0)
# reshape is redundant, but provide clearance
percentage = 100 * cal / A.reshape(1,4)
```
Notes:
- `axis=0` is vertical operation. So it will iterate over all **row** and doing operations on all **columns**.
- reshape command requires constant time.

General Principle of broadcasting
- if you have (m,n) matrix and do operation with (1, n), will results in (m,n)
- if you have (m,n) matrix and do operation with (m, 1), will results in (m,n) as well