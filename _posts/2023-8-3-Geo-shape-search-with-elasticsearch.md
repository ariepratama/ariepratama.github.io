---
layout: post
title: Region / geo shape search with elasticsearch
tags:
    - software engineering
    - elasticsearch
    - PostGIS
    - Geo Spatial
---
On my previous [article](https://ariepratama.github.io/Geo-search-with-elasticsearch/), I have demonstrated how to search for locations, which is basically points, given a region (in form of polygon) or central point. Now what if we have a bunch of regions and we want to query those regions, to answer the question of "given a point and a circle from that point, what are the regions that surround it?". 

# 1. Create index mapping
I will create similar index mapping, but this time, instead of using `geo_point`, since I would like to represent a region, I will have to use `geo_shape` type.

```json
PUT ny-clusters-001
{
  "mappings": {
    "properties": {
      "cluster": {"type": "geo_shape"},
      "cluster_name": {"type": "text"}
    }
  }
}
```

# 2. Insert documents
Again I shall re-draw maps with those regions. Here, I will pick New York City Hall, and draw 3 regions in this custom map.
<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1Z0_HDrZzZHvY_-hSpKPg774J1n4VaxY&ehbc=2E312F" width="640" height="480"></iframe>


you can see there are 3 gray regions, that I have drawn. My maps feature has enabled us to download the regions as CSV, with well-known text (WKT) format. This WKT could be imported directly into elasticsearch! I will demonstrate the api calls to populate those regions as below.
```json
POST ny-clusters-001/_doc
{
  "cluster_name": "NY City Hall",
  "cluster": "POLYGON ((-74.0080488 40.7120072, -74.0076089 40.7116819, -74.005608 40.7122105, -74.0051145 40.7123813, -74.0047819 40.7125521, -74.0042508 40.7131335, -74.0062947 40.7141257, -74.0080488 40.7120072))"
}

POST ny-clusters-001/_doc
{
  "cluster_name": "NY City Hall Cluster 1",
  "cluster": "POLYGON ((-74.0099852 40.712137, -74.0086549 40.7115352, -74.0063911 40.7141538, -74.0078931 40.7148532, -74.0099852 40.712137))"
}

POST ny-clusters-001/_doc
{
  "cluster_name": "NY City Hall Cluster 2",
  "cluster": "POLYGON ((-74.0048139 40.7124379, -74.0014129 40.7099168, -73.9983445 40.7132512, -74.0020995 40.715089, -74.0025072 40.7142921, -74.0033977 40.7136903, -74.0048139 40.7124379))"
}

```



# 3. Query by point and distance
Now I have my data, it's time to query those region. 


This first query will results in 2 regions: `NY City Hall` and `NY City Hall Cluster 1`, because those 2 regions are the only regions that could be intersected by "circle" within 71 meters from the query point. The query point and distance could be ilustrated as below.
![query1](/images/posts/2023-8-3-Geo-shape-search-with-elasticsearch/query1.png)

The sample request to search is as follow.
```json
GET ny-clusters-001/_search
{
  "query": {
    "geo_distance": {
      "distance": "71m",
      "cluster": {
        "lat": 40.71182,
        "lon": -74.00531
      }
    }
  }
}
```

This will return all the data that I have, since I have expanded the query "circle" to include all of the regions. Now the query region will look like this.
![query2](/images/posts/2023-8-3-Geo-shape-search-with-elasticsearch/query2.png)

And we will need to alter the `geo_distance`'s `distance`
```json
GET ny-clusters-001/_search
{
  "query": {
    "geo_distance": {
      "distance": "250m",
      "cluster": {
        "lat": 40.71182,
        "lon": -74.00531
      }
    }
  }
}
```


There you have it! I can now use this elasticsearch to query regions!


# References
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-geo-distance-query.html
- https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry

