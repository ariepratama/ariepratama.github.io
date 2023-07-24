---
layout: post
title: Geo spatial queries with elasticsearch
tags:
    - software engineering
    - elasticsearch
    - PostGIS
    - Geo Spatial
---
Other than [PostGIS](http://postgis.net/), [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/elasticsearch-intro.html) has the similar functionality to do geo spatial queries. Here I am not going to compare between these two technologies, but to my limited knowledge, elasticsearch could be used for geo spatial search, however it should not be used to store persistent data as opposed to PostgreSQL and PostGIS. I will focus instead to how to index geo spatial documents / data and query it.


# 1. Create index mapping
A first step to enable querying elasticsearch is always to enable the right data mapping. Here I will use geo_point for `location` field, that will give elasticsearch a clue that I am going to store `lat` and `lon` at this property. The other field is to store the `location_name` for the document.
```json
PUT geo-index-0001
{
  "mappings": {
    "properties": {
      "location": {"type": "geo_point"},
      "location_name": {"type": "text"}
    }
  }
}
```

# 2. Insert documents
Then I will fill in some dummy but real data, based on google maps. 

First I created a custom map on [my map feature](https://www.google.com/mymaps) from google maps. Then I draw a square polygon that will indicate the Time Square area that I wanted. The resulting map will look like this.
<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1ETfiwvSThHyyox6S62FEprcl60khA6A&ehbc=2E312F" width="640" height="480"></iframe>

Next I will populate some data to indicate landmarks, and I have chosen a few coffeshops around Times Square.
```json
POST geo-index-0001/_doc
{
  "location_name": "Time Square",
  "location": {
    "lat": 40.758049,
    "lon": -73.9878585
  }
}

POST geo-index-0001/_doc
{
  "location_name": "Val Cafe, Time Square",
  "location": {
    "lat": 40.75795,
    "lon": -73.9866843
  }
}

POST geo-index-0001/_doc
{
  "location_name": "Starbucks, Time Square",
  "location": {
    "lat": 40.7583137,
    "lon": -73.9868667
  }
}
```

and one location will be placed outside of Time Square area, so I can demonstrate the negative sample or sample that is supposed to be excluded when I query by geo polygon. 
```json
POST geo-index-0001/_doc
{
  "location_name": "Gregorys Coffee, New York",
  "location": {
    "lat": 40.7342684,
    "lon": -74.0147306
  }
}
```

# 3. Query by polygon
The old deprecated way to query elasticsearch by polygon is through `geo_polygon` json key. 
```json
GET geo-index-0001/_search
{
  "query": {
    "geo_polygon": {
      "location": {
        "points": [
          [-73.9880304, 40.7597602], 
          [-73.9898329, 40.7572491], 
          [-73.9841359, 40.7548109], 
          [-73.9822691, 40.7573791], 
          [-73.9880304, 40.7597602]
        ]
      }
    }
  }  
}
```

Alternatively, elasticsearch documentation recommended us to move to `geo_shape`. The `geo_shape` has the capability to use [Well-known text](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) representation to define polygon or other shapes. For example:

```json
GET geo-index-0001/_search
{
  "query": {
    "geo_shape": {
      "location": {
        "shape": "POLYGON ((-73.9880304 40.7597602, -73.9898329 40.7572491, -73.9841359 40.7548109, -73.9822691 40.7573791, -73.9880304 40.7597602))",
        "relation": "within"
      }
    }
  }  
}
```

The response, if you have followed by direction, will looks like this, where all entities around the polygon will be included except one that is outside the polygon. 
```json
...
  "hits" : {
    "total" : {
      "value" : 3,
      "relation" : "eq"
    },
    "max_score" : 0.0,
    "hits" : [
      {
        "_index" : "geo-index-0001",
        "_type" : "_doc",
        "_id" : "pO70bokBOz2vX6qDjobs",
        "_score" : 0.0,
        "_source" : {
          "location_name" : "Time Square",
          "location" : {
            "lat" : 40.758049,
            "lon" : -73.9878585
          }
        }
      },
      {
        "_index" : "geo-index-0001",
        "_type" : "_doc",
        "_id" : "pe71bokBOz2vX6qDdIZ5",
        "_score" : 0.0,
        "_source" : {
          "location_name" : "Val Cafe, Time Square",
          "location" : {
            "lat" : 40.75795,
            "lon" : -73.9866843
          }
        }
      },
      {
        "_index" : "geo-index-0001",
        "_type" : "_doc",
        "_id" : "pu72bokBOz2vX6qDgoYH",
        "_score" : 0.0,
        "_source" : {
          "location_name" : "Starbucks, Time Square",
          "location" : {
            "lat" : 40.7583137,
            "lon" : -73.9868667
          }
        }
      }
    ]
  }
...
```

# 4. Query by distance
One more feature to note is querying by distance from a point. Here I can use `geo_distance` key, and I will define the same `lat` and `lon` from `Times Square` coordinate. The query looks like this.
```json
GET geo-index-0001/_search
{
  "query": {
    "geo_distance": {
      "distance": "1km",
      "location": {
        "lat": 40.758049,
        "lon": -73.9878585 
      }
    }
  }
}
```



# References
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-geo-shape-query.html
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-geo-distance-query.html
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/mapping-types.html
- https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry

