---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Code Snippets For SpaCy

```python
%load_ext autoreload
%autoreload 2
```

```python
import spacy
```

```python
# Run this line if you haven just installed spacy
# !python -m spacy download en_core_web_sm
```

```python
import en_core_web_sm
nlp = en_core_web_sm.load()
```

```python
doc = nlp(
"Trump has long touted China’s huge exports to the U.S. "
"as a sign that Beijing has been taking advantage "
"of American businesses for decades."
)
```

## Named Entity Recognition

Upon loading a corpus, SpaCy has actively run functions like POS Tags and NER if a language model provides.

```python
doc.ents
```

show the NE labels

```python
[(ent, ent.label_) for ent in doc.ents]
```

## POS Tag

```python
list(doc.noun_chunks)
```
