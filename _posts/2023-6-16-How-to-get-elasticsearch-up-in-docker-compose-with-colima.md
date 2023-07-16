---
layout: post
title: How to get elasticsearch up in docker-compose with colima
tags:
    - software engineering
    - docker
    - colima
    - elasticsearch
---
[Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/elasticsearch-intro.html) is a great tool to enable search within, not only text data, but also multifield objects. [Colima](https://github.com/abiosoft/colima) is an alternative to docker, since docker desktop has changed their license to be more commercial focus I have to search for alternative and I found that Colima is one stable alternative for mac. In this guide I will introduce a quick start how to get elasticsearch up in local development environment with docker-compose and colima. 

# Configure colima
Elasticsearch requires certain configuration on the virtual machine itself, specifically `max_map_count` parameter. Here is how to do that with colima.

After you have installed Colima, you may do 

```colima start --edit```

or (yes I used visual code to do this, hence the command is `code [path]`)

```code ~/.colima/default/colima.yaml```

and add `provision` configuration

```yaml
...
provision:
  - mode: system
    script: sysctl -w vm.max_map_count=262144
...
```

# Get your docker-compose yaml ready
I will only use single node setup for elasticsearch 7, since I do not need more than one nodes in my local development. Feel free to see the full configuration [here](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/elasticsearch-intro.html).

```yaml
version: '2.2'
services:
  es01:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.11
    environment:
      - node.name=es01
      - cluster.name=es-docker-cluster
      - cluster.initial_master_nodes=es01
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - data01:/usr/share/elasticsearch/data
    network_mode: host

volumes:
  data01:
    driver: local
```


# Then you are done!
Then you are good to go! just need to do `docker-compose up` and your service should be up!


nb: if you had `exit code 137` that means you should expand your memory limit in colima.


# References
- https://www.elastic.co/guide/en/elasticsearch/reference/7.17/elasticsearch-intro.html
- https://github.com/abiosoft/colima/issues/384

