---
layout: post
title: Autocomplete search with elasticsearch
tags:
    - software engineering
    - docker
    - colima
    - elasticsearch
---
[Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/elasticsearch-intro.html) main functionality is not limited to text search with specific tokenizer / [analyzer](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/analysis-analyzers.html), but it would also be able to serve a fast autocomplete! Here I will briefly explain simple steps to enable the autocomplete search.

# 1. Create completion suggester index and mapping
[Completion suggester](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-suggesters.html#completion-suggester) is one of mapping type that elasticsearch has. it may handle prefix search and "autocorrect" functionality. 

You may open kibana -> dev tools, then create an index like this
```
PUT completion-index-0001
{
  "mappings": {
    "properties": {
      "text_suggest": {"type": "completion"},
      "text": {"type": "text"}
    }
  }
}
```
Here there are two fields:
* `text_suggest`: mapped to **completion** mapping. We will be using this mapping for our autocomplete.
* `text`: mapped to text mapping, which will use standard analyzer by default.
 
# 2. Put documents into the index
Puting documents into the index is no different than any other indexes. You may also put different text for suggestion than the original, but for simplicity purpose I will use the same text. here is the script to populate documents.

```
POST completion-index-0001/_doc
{
  "text": "I want to break free",
  "text_suggest": "I want to break free"
}

POST completion-index-0001/_doc
{
  "text": "work work work work work",
  "text_suggest": "work work work work work"
}

POST completion-index-0001/_doc
{
  "text": "bohemian Raphsody",
  "text_suggest": "bohemian Raphsody"
}
```
# 3. Query the index with "suggest" structure
## Autocomplete search
Now you can play around with the autocomplete mapping, the format of the request should be something like below.
```
POST [index-name]/_search
{
  "suggest": {
    [arbitrary-string]: {
      ["text" | "prefix"]: [input your text query to be autocompleted],
      "completion": {
        "field": [field with type "completion"]
      }
    }
  }
}
```

sample request
```
POST completion-index-0001/_search
{
  "suggest": {
    "x": {
      "text": ["i"],
      "completion": {
        "field": "text_suggest"
      } 
    }
    
  }
}
```

sample response 
```
{
  "took" : 3,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 0,
      "relation" : "eq"
    },
    "max_score" : null,
    "hits" : [ ]
  },
  "suggest" : {
    "x" : [
      {
        "text" : "i",
        "offset" : 0,
        "length" : 1,
        "options" : [
          {
            "text" : "I want to break free",
            "_index" : "completion-index-0001",
            "_type" : "_doc",
            "_id" : "IFCsZokBMQ4penAGtHni",
            "_score" : 1.0,
            "_source" : {
              "text" : "I want to break free",
              "text_suggest" : "I want to break free"
            }
          }
        ]
      }
    ]
  }
}
```

Note that, this index will not be able to search **keywords in between**. Thus if you search for `want`, it will not return any result, for example this request
```
POST completion-index-0001/_search
{
  "suggest": {
    "x": {
      "text": ["want"],
      "completion": {
        "field": "text_suggest"
      } 
    }
    
  }
}
```
will return 0 results.

## Fuzzy search
`completion` mapping will enable the option to do fuzzy search / in layman term, autocorrect search. You may input some typo, but the service will still be able to return the result.

Consider this example where I had a typo, from `work`, to `wrk`, for example

```
POST completion-index-0001/_search
{
  "suggest": {
    "x": {
      "text": ["wrk"],
      "completion": {
        "field": "text_suggest",
        "fuzzy": {
          "fuzziness": 2
        }
      } 
    }
    
  }
}
```

sample response 
```
...
"suggest" : {
    "x" : [
      {
        "text" : "wrk",
        "offset" : 0,
        "length" : 3,
        "options" : [
          {
            "text" : "work work work work work",
            "_index" : "completion-index-0001",
            "_type" : "_doc",
            "_id" : "eLatZokBv2x7mLzMBsp-",
            "_score" : 1.0,
            "_source" : {
              "text" : "work work work work work",
              "text_suggest" : "work work work work work"
            }
          }
        ]
      }
    ]
  }
```


The elasticsearch `completion` index uses [Finite State Transducer](https://www.learningstuffwithankit.dev/implementing-auto-complete-functionality-in-elasticsearch-part-iii-completion-suggester), which explained why it is not suitable to search keyword in between. 


That is all for this article, I would like this article to be bite-sized! Full elasticsearch script can be found at my [github](https://github.com/ariepratama/go-playground/blob/main/es/es-autocomplete-sample-script.md). While the docker-compose file can be found on another [path](https://github.com/ariepratama/go-playground/blob/main/es/docker-compose.yml).




# References
- https://www.elastic.co/videos/using-the-completion-suggester 
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/mapping-types.html
- https://www.learningstuffwithankit.dev/implementing-auto-complete-functionality-in-elasticsearch-part-iii-completion-suggester
