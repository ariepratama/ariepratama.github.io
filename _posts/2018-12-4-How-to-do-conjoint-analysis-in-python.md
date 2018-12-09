---
layout: post
title: How to Do Conjoint Analysis in python
tags:
    - statistics
---
Conjoint analysis is a method to find the most prefered settings of a product [11].

Usual fields of usage [3]:
- Marketing
- Product management
- Operation Research

For example:
- testing customer acceptance of new product design.
- assessing appeal of advertisements and service design.


```python
import pandas as pd
import numpy as np
```

Here we used Immigrant conjoint data described by [6]. It consists of 2 possible conjoint methods: choice-based conjoint (with `selected` column as target variable) and rating-based conjoint (with `rating` as target variable). 

# Preparing The Data


```python
# taken from imigrant conjoint data
df = pd.read_csv('https://dataverse.harvard.edu/api/access/datafile/2445996?format=tab&gbrecs=true', delimiter='\t')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resID</th>
      <th>atmilitary</th>
      <th>atreligion</th>
      <th>ated</th>
      <th>atprof</th>
      <th>atinc</th>
      <th>atrace</th>
      <th>atage</th>
      <th>atmale</th>
      <th>selected</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>383</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>383</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>383</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>383</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking for empty data
df.isnull().sum()
```




    resID          0
    atmilitary     0
    atreligion     0
    ated           0
    atprof         0
    atinc          0
    atrace         0
    atage          0
    atmale         0
    selected       0
    rating        10
    dtype: int64




```python
# remove empty data
clean_df = df[~df.rating.isnull()]
```

# Doing The Conjoint Analysis


```python

y = clean_df['selected']
x = clean_df[[x for x in df.columns if x != 'selected' and x != 'resID' and x != 'rating']]
```


```python
xdum = pd.get_dummies(x, columns=[c for c in x.columns if c != 'selected'])
xdum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atmilitary_1</th>
      <th>atmilitary_2</th>
      <th>atreligion_1</th>
      <th>atreligion_2</th>
      <th>atreligion_3</th>
      <th>atreligion_4</th>
      <th>atreligion_5</th>
      <th>atreligion_6</th>
      <th>ated_1</th>
      <th>ated_2</th>
      <th>...</th>
      <th>atrace_5</th>
      <th>atrace_6</th>
      <th>atage_1</th>
      <th>atage_2</th>
      <th>atage_3</th>
      <th>atage_4</th>
      <th>atage_5</th>
      <th>atage_6</th>
      <th>atmale_1</th>
      <th>atmale_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



[11] has complete definition of important attributes in Conjoint Analysis


**Utility of an alternative $U(x)$** is

$$
U(x) = \sum_{i=1}^{m}\sum_{j=1}^{k_{i}}u_{ij}x_{ij}
$$

where:

$u_{ij}$: part-worth contribution (utility of jth level of ith attribute)

$k_{i}$: number of levels for attribute i

$m$: number of attributes


**Importance of an attribute $R_{i}$** is defined as
$$
R_{i} = max(u_{ij})  - min(u_{ik})
$$
$R_{i}$ is the $i$-th attribute


**Relative Importance of an attribute $Rimp_{i}$** is defined as
$$
Rimp_{i} = \frac{R_{i}}{\sum_{i=1}^{m}{R_{i}}}
$$

Essentially conjoint analysis (traditional conjoint analysis) is doing **linear regression** where the target variable could be binary (**choice-based conjoint analysis**), or 1-7 likert scale (**rating conjoint analysis**), or ranking(**rank-based conjoint analysis**).



```python
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
```


```python
res = sm.OLS(y, xdum, family=sm.families.Binomial()).fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>selected</td>     <th>  R-squared:         </th> <td>   0.091</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.083</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   10.72</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 09 Dec 2018</td> <th>  Prob (F-statistic):</th> <td>7.39e-51</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:49:37</td>     <th>  Log-Likelihood:    </th> <td> -2343.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3456</td>      <th>  AIC:               </th> <td>   4753.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3423</td>      <th>  BIC:               </th> <td>   4956.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    32</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>atmilitary_1</th> <td>    0.0808</td> <td>    0.008</td> <td>    9.585</td> <td> 0.000</td> <td>    0.064</td> <td>    0.097</td>
</tr>
<tr>
  <th>atmilitary_2</th> <td>    0.1671</td> <td>    0.008</td> <td>   19.810</td> <td> 0.000</td> <td>    0.151</td> <td>    0.184</td>
</tr>
<tr>
  <th>atreligion_1</th> <td>    0.0931</td> <td>    0.018</td> <td>    5.132</td> <td> 0.000</td> <td>    0.058</td> <td>    0.129</td>
</tr>
<tr>
  <th>atreligion_2</th> <td>    0.0578</td> <td>    0.018</td> <td>    3.144</td> <td> 0.002</td> <td>    0.022</td> <td>    0.094</td>
</tr>
<tr>
  <th>atreligion_3</th> <td>    0.0803</td> <td>    0.018</td> <td>    4.411</td> <td> 0.000</td> <td>    0.045</td> <td>    0.116</td>
</tr>
<tr>
  <th>atreligion_4</th> <td>    0.0797</td> <td>    0.018</td> <td>    4.326</td> <td> 0.000</td> <td>    0.044</td> <td>    0.116</td>
</tr>
<tr>
  <th>atreligion_5</th> <td>   -0.0218</td> <td>    0.018</td> <td>   -1.185</td> <td> 0.236</td> <td>   -0.058</td> <td>    0.014</td>
</tr>
<tr>
  <th>atreligion_6</th> <td>   -0.0411</td> <td>    0.018</td> <td>   -2.256</td> <td> 0.024</td> <td>   -0.077</td> <td>   -0.005</td>
</tr>
<tr>
  <th>ated_1</th>       <td>   -0.1124</td> <td>    0.018</td> <td>   -6.115</td> <td> 0.000</td> <td>   -0.148</td> <td>   -0.076</td>
</tr>
<tr>
  <th>ated_2</th>       <td>    0.0278</td> <td>    0.019</td> <td>    1.464</td> <td> 0.143</td> <td>   -0.009</td> <td>    0.065</td>
</tr>
<tr>
  <th>ated_3</th>       <td>    0.0366</td> <td>    0.019</td> <td>    1.942</td> <td> 0.052</td> <td>   -0.000</td> <td>    0.074</td>
</tr>
<tr>
  <th>ated_4</th>       <td>    0.0737</td> <td>    0.018</td> <td>    4.076</td> <td> 0.000</td> <td>    0.038</td> <td>    0.109</td>
</tr>
<tr>
  <th>ated_5</th>       <td>    0.0649</td> <td>    0.018</td> <td>    3.570</td> <td> 0.000</td> <td>    0.029</td> <td>    0.101</td>
</tr>
<tr>
  <th>ated_6</th>       <td>    0.1572</td> <td>    0.018</td> <td>    8.949</td> <td> 0.000</td> <td>    0.123</td> <td>    0.192</td>
</tr>
<tr>
  <th>atprof_1</th>     <td>    0.1084</td> <td>    0.018</td> <td>    5.930</td> <td> 0.000</td> <td>    0.073</td> <td>    0.144</td>
</tr>
<tr>
  <th>atprof_2</th>     <td>    0.0852</td> <td>    0.019</td> <td>    4.597</td> <td> 0.000</td> <td>    0.049</td> <td>    0.122</td>
</tr>
<tr>
  <th>atprof_3</th>     <td>    0.0910</td> <td>    0.018</td> <td>    5.060</td> <td> 0.000</td> <td>    0.056</td> <td>    0.126</td>
</tr>
<tr>
  <th>atprof_4</th>     <td>    0.0674</td> <td>    0.018</td> <td>    3.716</td> <td> 0.000</td> <td>    0.032</td> <td>    0.103</td>
</tr>
<tr>
  <th>atprof_5</th>     <td>    0.0145</td> <td>    0.019</td> <td>    0.779</td> <td> 0.436</td> <td>   -0.022</td> <td>    0.051</td>
</tr>
<tr>
  <th>atprof_6</th>     <td>   -0.1186</td> <td>    0.018</td> <td>   -6.465</td> <td> 0.000</td> <td>   -0.155</td> <td>   -0.083</td>
</tr>
<tr>
  <th>atinc_1</th>      <td>    0.0081</td> <td>    0.018</td> <td>    0.448</td> <td> 0.654</td> <td>   -0.027</td> <td>    0.043</td>
</tr>
<tr>
  <th>atinc_2</th>      <td>    0.0316</td> <td>    0.019</td> <td>    1.662</td> <td> 0.097</td> <td>   -0.006</td> <td>    0.069</td>
</tr>
<tr>
  <th>atinc_3</th>      <td>    0.0716</td> <td>    0.018</td> <td>    4.020</td> <td> 0.000</td> <td>    0.037</td> <td>    0.106</td>
</tr>
<tr>
  <th>atinc_4</th>      <td>    0.0397</td> <td>    0.018</td> <td>    2.154</td> <td> 0.031</td> <td>    0.004</td> <td>    0.076</td>
</tr>
<tr>
  <th>atinc_5</th>      <td>    0.0808</td> <td>    0.018</td> <td>    4.451</td> <td> 0.000</td> <td>    0.045</td> <td>    0.116</td>
</tr>
<tr>
  <th>atinc_6</th>      <td>    0.0161</td> <td>    0.018</td> <td>    0.872</td> <td> 0.383</td> <td>   -0.020</td> <td>    0.052</td>
</tr>
<tr>
  <th>atrace_1</th>     <td>    0.0274</td> <td>    0.018</td> <td>    1.494</td> <td> 0.135</td> <td>   -0.009</td> <td>    0.063</td>
</tr>
<tr>
  <th>atrace_2</th>     <td>    0.0527</td> <td>    0.018</td> <td>    2.881</td> <td> 0.004</td> <td>    0.017</td> <td>    0.089</td>
</tr>
<tr>
  <th>atrace_3</th>     <td>    0.0633</td> <td>    0.018</td> <td>    3.556</td> <td> 0.000</td> <td>    0.028</td> <td>    0.098</td>
</tr>
<tr>
  <th>atrace_4</th>     <td>    0.0037</td> <td>    0.019</td> <td>    0.198</td> <td> 0.843</td> <td>   -0.033</td> <td>    0.040</td>
</tr>
<tr>
  <th>atrace_5</th>     <td>    0.0324</td> <td>    0.018</td> <td>    1.787</td> <td> 0.074</td> <td>   -0.003</td> <td>    0.068</td>
</tr>
<tr>
  <th>atrace_6</th>     <td>    0.0683</td> <td>    0.019</td> <td>    3.687</td> <td> 0.000</td> <td>    0.032</td> <td>    0.105</td>
</tr>
<tr>
  <th>atage_1</th>      <td>    0.0680</td> <td>    0.018</td> <td>    3.770</td> <td> 0.000</td> <td>    0.033</td> <td>    0.103</td>
</tr>
<tr>
  <th>atage_2</th>      <td>    0.0934</td> <td>    0.019</td> <td>    4.957</td> <td> 0.000</td> <td>    0.056</td> <td>    0.130</td>
</tr>
<tr>
  <th>atage_3</th>      <td>    0.0900</td> <td>    0.018</td> <td>    4.967</td> <td> 0.000</td> <td>    0.054</td> <td>    0.125</td>
</tr>
<tr>
  <th>atage_4</th>      <td>    0.0711</td> <td>    0.019</td> <td>    3.837</td> <td> 0.000</td> <td>    0.035</td> <td>    0.107</td>
</tr>
<tr>
  <th>atage_5</th>      <td>    0.0038</td> <td>    0.018</td> <td>    0.208</td> <td> 0.835</td> <td>   -0.032</td> <td>    0.039</td>
</tr>
<tr>
  <th>atage_6</th>      <td>   -0.0783</td> <td>    0.018</td> <td>   -4.276</td> <td> 0.000</td> <td>   -0.114</td> <td>   -0.042</td>
</tr>
<tr>
  <th>atmale_1</th>     <td>    0.1228</td> <td>    0.008</td> <td>   14.616</td> <td> 0.000</td> <td>    0.106</td> <td>    0.139</td>
</tr>
<tr>
  <th>atmale_2</th>     <td>    0.1250</td> <td>    0.008</td> <td>   14.787</td> <td> 0.000</td> <td>    0.108</td> <td>    0.142</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.070</td> <th>  Durbin-Watson:     </th> <td>   2.872</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.966</td> <th>  Jarque-Bera (JB):  </th> <td> 391.306</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.011</td> <th>  Prob(JB):          </th> <td>1.07e-85</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.352</td> <th>  Cond. No.          </th> <td>1.27e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 4.28e-29. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
df_res = pd.DataFrame({
    'param_name': res.params.keys()
    , 'param_w': res.params.values
    , 'pval': res.pvalues
})
# adding field for absolute of parameters
df_res['abs_param_w'] = np.abs(df_res['param_w'])
# marking field is significant under 95% confidence interval
df_res['is_sig_95'] = (df_res['pval'] < 0.05)
# constructing color naming for each param
df_res['c'] = ['blue' if x else 'red' for x in df_res['is_sig_95']]

# make it sorted by abs of parameter value
df_res = df_res.sort_values(by='abs_param_w', ascending=True)
```


```python
f, ax = plt.subplots(figsize=(14, 8))
plt.title('Part Worth')
pwu = df_res['param_w']
xbar = np.arange(len(pwu))
plt.barh(xbar, pwu, color=df_res['c'])
plt.yticks(xbar, labels=df_res['param_name'])
plt.show()
```


![png](/images/posts/2018-12-4-How-to-do-conjoint-analysis/output_16_0.png)


Now we will compute importance of every attributes, with definition from before, where:


$$
R_{i} = max(u_{ij})  - min(u_{ik})
$$




$$
Rimp_{i} = \frac{R_{i}}{\sum_{i=1}^{m}{R_{i}}}
$$


sum of importance on attributes will approximately equal to the target variable scale: if it is choice-based then it will equal to 1, if it is likert scale 1-7 it will equal to 7. In this case, importance of an attribute will equal with relative importance of an attribute because it is choice-based conjoint analysis (the target variable is binary).


```python
# need to assemble per attribute for every level of that attribute in dicionary
range_per_feature = dict()
for key, coeff in res.params.items():
    sk =  key.split('_')
    feature = sk[0]
    if len(sk) == 1:
        feature = key
    if feature not in range_per_feature:
        range_per_feature[feature] = list()
        
    range_per_feature[feature].append(coeff)
```


```python
# importance per feature is range of coef in a feature
# while range is simply max(x) - min(x)
importance_per_feature = {
    k: max(v) - min(v) for k, v in range_per_feature.items()
}

# compute relative importance per feature
# or normalized feature importance by dividing 
# sum of importance for all features
total_feature_importance = sum(importance_per_feature.values())
relative_importance_per_feature = {
    k: 100 * round(v/total_feature_importance, 3) for k, v in importance_per_feature.items()
}

```


```python
alt_data = pd.DataFrame(
    list(importance_per_feature.items()), 
    columns=['attr', 'importance']
).sort_values(by='importance', ascending=False)


f, ax = plt.subplots(figsize=(12, 8))
xbar = np.arange(len(alt_data['attr']))
plt.title('Importance')
plt.barh(xbar, alt_data['importance'])
for i, v in enumerate(alt_data['importance']):
    ax.text(v , i + .25, '{:.2f}'.format(v))
plt.ylabel('attributes')
plt.xlabel('% importance')
plt.yticks(xbar, alt_data['attr'])
plt.show()
```


![png](/images/posts/2018-12-4-How-to-do-conjoint-analysis/output_20_0.png)



```python
alt_data = pd.DataFrame(
    list(relative_importance_per_feature.items()), 
    columns=['attr', 'relative_importance (pct)']
).sort_values(by='relative_importance (pct)', ascending=False)


f, ax = plt.subplots(figsize=(12, 8))
xbar = np.arange(len(alt_data['attr']))
plt.title('Relative importance / Normalized importance')
plt.barh(xbar, alt_data['relative_importance (pct)'])
for i, v in enumerate(alt_data['relative_importance (pct)']):
    ax.text(v , i + .25, '{:.2f}%'.format(v))
plt.ylabel('attributes')
plt.xlabel('% relative importance')
plt.yticks(xbar, alt_data['attr'])
plt.show()
```


![png](/images/posts/2018-12-4-How-to-do-conjoint-analysis/output_21_0.png)


# References
[1] [Lifestyles on Github](https://github.com/CamDavidsonPilon/lifestyles)

[2] [Hierarchical Bayes](http://webuser.bus.umich.edu/plenk/HB%20Conjoint%20Lenk%20DeSarbo%20Green%20Young%20MS%201996.pdf)

[3] [Conjoint Analysis Wikipedia](https://en.wikipedia.org/wiki/Conjoint_analysis)

[4] [Conjoint Analysis - Towards Data Science Medium](https://towardsdatascience.com/conjoint-analysis-101-7bf5dfe1cdb2)

[5] [Hainmueller, Jens;Hopkins, Daniel J.;Yamamoto, Teppei, 2013, "Replication data for: Causal Inference in Conjoint Analysis: Understanding Multidimensional Choices via Stated Preference Experiments"](https://hdl.handle.net/1902.1/22603)

[6] [Causal Inference in Conjoint Analysis: Understanding
Multidimensional Choices via Stated Preference Experiments](http://web.mit.edu/teppei/www/research/conjoint.pdf)


[8] [Traditional Conjoin Analysis - Jupyter Notebook](https://github.com/Herka/Traditional-Conjoint-Analysis-with-Python/blob/master/Traditional%20Conjoint%20Analyse.ipynb)

[9] [Business Research Method - 2nd Edition - Chap 19](https://www.safaribooksonline.com/library/view/business-research-methods/9789352861620/xhtml/Chapter19.xhtml)

[10] [Tentang Data - Conjoint Analysis Part 1 (Bahasa Indonesia)](https://tentangdata.wordpress.com/2018/05/08/petunjuk-perancangan-dan-analisis-dalam-survei-conjoint-analysis-bag-1/)

[11] [Business Research Method, 2nd Edition, Chapter 19 (Safari Book Online)](https://www.safaribooksonline.com/library/view/business-research-methods/9789352861620/xhtml/Chapter19.xhtml)

