---
layout: post
title: How to Use One Way ANOVA in Python
---

One way ANOVA (Analysis of Variance) is a technique for hypothesis testing. It is used to test whether the means of different group is really different. 

Okaaaaay.... 

But then for all of you that are not used with **Statistics**, there might be big question arise: "Why would we even need to test this hypothesis testing?"

![but_why](https://media.giphy.com/media/1M9fmo1WAFVK0/giphy.gif)

Let me explain really quickly. In real world, most of the time we can only take samples of quantitative data. Say that I conduct a simple survey, I want to know the height of people in a town with population 10 million, and I could only ask 100 people.

Now, I take the first survey and got 100 people data, it shows that the average of people's height in the town are 170cm for men and 165cm for women. But somehow I want to re-do the survey in next day


Then I re-do the survey (assumes that I don't have the same person in with previous survey) and suprisingly it shows that the average of people's height in the town are 165cm for men and 165cm for women. Okay, now I am confused. 

![what](https://media.giphy.com/media/SqmkZ5IdwzTP2/giphy.gif)

Sooooo.. do men and women have same height in this town? 

This is it! It's time to use statistics that you all have learnt!



```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np
```

    /home/arie/miniconda2/envs/data_analysis_351/lib/python3.5/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
# don't ask about this, it's just happens to be a good style!
plt.style.use('fivethirtyeight')
```

## The Hypothesis
For this toy problem purpose, I have a hypothesis that
> for each diets, people weight's mean is same.



## Load The Data

Here I am using the **Diet Dataset** (see [here](https://www.sheffield.ac.uk/mash/statistics2/anova) for more datasets) from University of Sheffield for this practice problem. From the description [here](https://bioinformatics-core-shared-training.github.io/linear-models-r/r-recap.nb.html), the gender is binary variable which contains 0 for Female and 1 for Male. 


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
plt.title('Weight Distributions Between Sample')
plt.ylabel('pdf')
sns.distplot(data.weight6weeks)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f744d471320>




![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_18_1.png)



```python
f, ax = plt.subplots(figsize=(11,9))
sns.distplot(data[data.gender == '1'].weight6weeks, ax=ax, label='Male')
sns.distplot(data[data.gender == '0'].weight6weeks, ax=ax, label='Female')
plt.title('Weight Distribution for Each Gender')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f744d5dee48>




![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_19_1.png)



```python
def infer_gender(x):
    if x == '1': 
        return 'Male'
    
    if x == '0':
        return 'Female'
    
    return 'Other'

def show_distribution(df, gender, column, group):
    f, ax = plt.subplots(figsize=(11,9))
    plt.title('Weight Distribution for {} on each {}'.format(gender, column))
    for group_member in group:
        sns.distplot(df[df[column] == group_member].weight6weeks, label='{}'.format(group_member))
    plt.legend()
    plt.show()
    
unique_diet = data.Diet.unique()
unique_gender = data.gender.unique()

for gender in unique_gender:
    if gender != ' ':
        show_distribution(data[data.gender == gender], infer_gender(gender), 'Diet', unique_diet)

```


![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_20_0.png)



![png](/images/posts/2018-6-7-How-to-Use-1-Way-Anova-in-Python/output_20_1.png)



```python
data.groupby('gender').agg(
    [np.mean, np.median, np.count_nonzero, np.std]
).weight6weeks
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
      <td>81.500000</td>
      <td>81.5</td>
      <td>2.0</td>
      <td>30.405592</td>
    </tr>
    <tr>
      <th>0</th>
      <td>63.223256</td>
      <td>62.4</td>
      <td>43.0</td>
      <td>6.150874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75.015152</td>
      <td>73.9</td>
      <td>33.0</td>
      <td>4.629398</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby(['gender', 'Diet']).agg(
    [np.mean, np.median, np.count_nonzero, np.std]
).weight6weeks
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
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>count_nonzero</th>
      <th>std</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>Diet</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <th>2</th>
      <td>81.500000</td>
      <td>81.50</td>
      <td>2.0</td>
      <td>30.405592</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>1</th>
      <td>64.878571</td>
      <td>64.50</td>
      <td>14.0</td>
      <td>6.877296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62.178571</td>
      <td>61.15</td>
      <td>14.0</td>
      <td>6.274635</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62.653333</td>
      <td>61.80</td>
      <td>15.0</td>
      <td>5.370537</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>1</th>
      <td>76.150000</td>
      <td>75.75</td>
      <td>10.0</td>
      <td>5.439414</td>
    </tr>
    <tr>
      <th>2</th>
      <td>73.163636</td>
      <td>72.70</td>
      <td>11.0</td>
      <td>3.818448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75.766667</td>
      <td>76.35</td>
      <td>12.0</td>
      <td>4.434848</td>
    </tr>
  </tbody>
</table>
</div>



We can see difference in weight on females in diet, but interestingly, it does not seems to affect males. LOL

![expected](https://media.giphy.com/media/L20mbc7yRfsly/giphy.gif)



**Next question is: will the population behave the same?**


## The 1 Way Anova

The 1 way anova's null hypothesis is
$$\mu_{weight_{diet1}} = \mu_{weight_{diet2}} = \mu_{weight_{diet3}}$$

and this tests tries to see if it is true or not true

let's assume that we have initially determine our confidence level of 95%, which means that we will accept 5% error rate.


```python
mod = ols('Height ~ Diet', data=data[data.gender=='0']).fit()
# do type 2 anova
aov_table = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for Female')
print('----------------------')
print(aov_table)
print()


mod = ols('Height ~ Diet', data=data[data.gender=='1']).fit()
# do type 2 anova
aov_table = sm.stats.anova_lm(mod, typ=2)
print('ANOVA table for Male')
print('----------------------')
print(aov_table)
```

    ANOVA table for Female
    ----------------------
                   sum_sq    df        F    PR(>F)
    Diet       559.680764   1.0  7.17969  0.010566
    Residual  3196.086677  41.0      NaN       NaN
    
    ANOVA table for Male
    ----------------------
                   sum_sq    df        F    PR(>F)
    Diet        67.801603   1.0  0.43841  0.512784
    Residual  4794.259003  31.0      NaN       NaN


There are two p-values(PR(>F)) that we can see here, male and female. 

For male, we cannot accept the null hypothesis under 95% confident level, because the p-value is greater than our alpha (0.05 < 0.512784). So given these three type of diet, there are no difference in male weights.

![i_know](https://media.giphy.com/media/QzKtmrdMw6Tra/giphy.gif)


For female, since the p-value PR(>F) is less than our error rate (0.05 > 0.010566), we could reject the null hypothesis. This means we are quite confident that **there is a different in height for Female in diets**. 

Okay so we know the effect of diet in female, but we don't know which diet is different from which. We have to do post-hoc analysis using Tukey HSD (Honest Significant Difference) Test.


```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
```


```python
# Only use female data
df = data[data.gender=='0']

# compare the height between each diet, using 95% confidence interval 
mc = MultiComparison(df['Height'], df['Diet'])
tukey_result = mc.tukeyhsd(alpha=0.05)

print(tukey_result)
print('Unique diet groups: {}'.format(mc.groupsunique))
```

    Multiple Comparison of Means - Tukey HSD,FWER=0.05
    ==============================================
    group1 group2 meandiff  lower    upper  reject
    ----------------------------------------------
      1      2    -3.5714  -11.7861  4.6432 False 
      1      3    -8.7714  -16.848  -0.6948  True 
      2      3      -5.2   -13.2766  2.8766 False 
    ----------------------------------------------
    Unique diet groups: [1 2 3]


We can only reject the null hypothesis between diet type 1 and diet type 3, means there is statistically significant difference in weight for diet 1 and diet 3.

# References:
- https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
- http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-16_23.html
- http://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
- https://www.sheffield.ac.uk/mash/statistics2/anova
- https://stackoverflow.com/questions/16049552/what-statistics-module-for-python-supports-one-way-anova-with-post-hoc-tests-tu

