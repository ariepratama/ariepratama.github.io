One way anova is a technique for hypothesis testing

```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np
```


```python
plt.style.use('fivethirtyeight')
```

## Load The Data

Here I am using the **Diet Dataset** (see [here](https://www.sheffield.ac.uk/mash/statistics2/anova) for more datasets) from University of Sheffield for this practice problem. From the description [here](https://bioinformatics-core-shared-training.github.io/linear-models-r/r-recap.nb.html), the gender is binary variable which contains 0 for Female and 1 for Male. For this toy problem purpose, I have a hypothesis that Female and Male Height's mean is same.


```python
data = pd.read_csv('https://www.sheffield.ac.uk/polopoly_fs/1.570199!/file/stcp-Rdataset-Diet.csv')
```

## Getting Sense of The Dataset


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Person</th>
      <th>gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>pre.weight</th>
      <th>Diet</th>
      <th>weight6weeks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td></td>
      <td>41</td>
      <td>171</td>
      <td>60</td>
      <td>2</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td></td>
      <td>32</td>
      <td>174</td>
      <td>103</td>
      <td>2</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>159</td>
      <td>58</td>
      <td>1</td>
      <td>54.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>46</td>
      <td>192</td>
      <td>60</td>
      <td>1</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>55</td>
      <td>170</td>
      <td>64</td>
      <td>1</td>
      <td>63.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('This dataset contains {} rows'.format(data.size))
```

    This dataset contains 546 rows


## See If There is Any Missing Values


```python
data.gender.unique()
```




    array([' ', '0', '1'], dtype=object)




```python
# show which person have missing value in gender
data[data.gender == ' ']
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Person</th>
      <th>gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>pre.weight</th>
      <th>Diet</th>
      <th>weight6weeks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td></td>
      <td>41</td>
      <td>171</td>
      <td>60</td>
      <td>2</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td></td>
      <td>32</td>
      <td>174</td>
      <td>103</td>
      <td>2</td>
      <td>103.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Missing value percentage of all data: {:.2f}%'.format(data[data.gender == ' '].size / data.size * 100))
```

    Missing value percentage of all data: 2.56%


Cool! we only have ~3% missing value, either we could ignore, delete, or classify it's gender by using the closest Height mean.

## Getting the Sense of the Height Distribution


```python
f, ax = plt.subplots(figsize=(11,9))
plt.title('Height Distributions Between Sample')
plt.ylabel('pdf')
sns.distplot(data.Height)
```








![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_13_1.png)



```python
f, ax = plt.subplots(figsize=(11,9))
sns.distplot(data[data.gender == '1'].Height, ax=ax, label='gender == Male')
sns.distplot(data[data.gender == '0'].Height, ax=ax, label='gender == Female')
plt.title('Height Distribution for Each Gender')
plt.legend()
```








![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_14_1.png)



```python
data.groupby('gender').agg([np.mean, np.median, np.count_nonzero, np.std]).Height
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>count_nonzero</th>
      <th>std</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <td>172.500000</td>
      <td>172.5</td>
      <td>2</td>
      <td>2.121320</td>
    </tr>
    <tr>
      <th>0</th>
      <td>167.348837</td>
      <td>169.0</td>
      <td>43</td>
      <td>9.456375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>175.242424</td>
      <td>175.0</td>
      <td>33</td>
      <td>12.326370</td>
    </tr>
  </tbody>
</table>
</div>



Although the Height Distribution for Each Gender shows that most of the Females have both mean and median below Males, it is also not clear that are all **males have a tendency to be taller than females? **

## The 1 Way Anova

The 1 way anova's null hypothesis is
$$\mu_{gender_1} = \mu_{gender_2}$$

and this tests tries to see if it is true or not true

let's assume that we have initially determine our confidence level of 99%, which means that we will accept 1% error rate.


```python
mod = ols('Height ~ gender', data=data).fit()
# do type 2 anova
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
```

                   sum_sq    df         F    PR(>F)
    gender    1169.159132   2.0  5.084876  0.008494
    Residual  8622.328048  75.0       NaN       NaN


Since the p-value $PR(>F)$ is less than our error rate (0.01 > 0.0085), we could reject the null hypothesis. This means we are quite confident that **there is a different in height for each gender**

# References:
- https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
- http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-16_23.html
- http://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
- https://www.sheffield.ac.uk/mash/statistics2/anova

