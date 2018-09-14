---
layout: post
title: Summary of Statistics Terms
tags:
    - statistics
---

In this post, I just want to summarize statistics terms, that might be used when analyzing data or reading papers.


# Hypothesis Testing
In general, hypothesis testing is: **comparing means**!

\*\*\* run away \*\*\*


![run_away](https://media.giphy.com/media/l0NwF1dnk7GRz3pK0/giphy.gif)

Well it's oversimplification, but most of them tries to prove whether or not means between two segments of population are indeed different. There are a couple of ways on how to do this, I will simply summarize them for now:

|Terms|Short Definition|
|---|---|
|ANOVA|Analysis of Variance|
|ANCOVA| Analysis of Covariance|
|t-test| Student test |
|MANOVA| Multivariate Analysis of Variance|
|MANCOVA| Multivariate Analysis of Covariance |


## ANOVA
Is a way to test difference in variance between two or more levels/factors/variants/treatments.

### One-Way
ANOVA with 1 independent variable and 2 treatments.

### Two-Way
ANOVA with 2 or more independent variables and 2 or more treatments.

## ANCOVA
Is an extension of ANOVA, by taking into account another variable that might covariated/correlated ([see difference between covariance and correlation](https://en.wikipedia.org/wiki/Covariance_and_correlation)) with dependent variable. 

Another way to explain<sup>7</sup>: ANCOVA evaluates if the means of dependent variable are equal across levels of treatment, while controling continuous nuisance variable.

## When To Use?
If you believe that the effect of some variables depends on other variables<sup>8</sup>.

## Why Use ANCOVA?
3 general application for using ANCOVA<sup>6</sup>:
1. Increasing power of F-test.
2. Equating Non-Equivalent Groups.
3. Means Adjustment for Multiple Dependent Variables.

## How to Conduct ANCOVA?
know the assumptions in ANCOVA!

5 Assumption in using ANCOVA<sup>7</sup>:
1. **Linear relationship between dependent variable and nuisance variable (or [concomitant variable](http://www.statisticshowto.com/concomitant-variable/))**.
2. **Homogenuous Error Variances**.

    The error is random
3. **Independence of Error Terms**.

    Errors is uncorrelated or in other words covariance matrix in diagonal.
4. **Normality of Error**.

    The error should be normally distributed, $$\epsilon \sim \mathcal{N}(0,\sigma^2)$$
5. **Homogeneity of regression slopes**

    In other words regression lines should be parallel among groups.


How to conduct ANCOVA:
1. Test multicollinearity.
2. Test the homogeneity of variance assumption: use [Levene's Test](https://en.wikipedia.org/wiki/Levene%27s_test).
3. Test the homogeneity of regression slopes assumption.
4. Run ANCOVA analysis.


# References
\[1] [https://keydifferences.com/difference-between-t-test-and-anova.html](https://keydifferences.com/difference-between-t-test-and-anova.html)

\[2] [https://jessicaaro.wordpress.com/2012/03/25/why-use-the-anova-over-a-t-test/](https://jessicaaro.wordpress.com/2012/03/25/why-use-the-anova-over-a-t-test/)

\[3] [http://www.statisticshowto.com/levels-in-statistics/](http://www.statisticshowto.com/levels-in-statistics/)

\[4]  [http://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/](http://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/)

\[5] [https://onlinecourses.science.psu.edu/stat502/node/183/](https://onlinecourses.science.psu.edu/stat502/node/183/)

\[6] [http://www.statisticssolutions.com/general-uses-of-analysis-of-covariance-ancova/](http://www.statisticssolutions.com/general-uses-of-analysis-of-covariance-ancova/)

\[7] [https://en.wikipedia.org/wiki/Analysis_of_covariance](https://en.wikipedia.org/wiki/Analysis_of_covariance)

\[8] [https://stats.stackexchange.com/questions/24077/how-to-choose-between-anova-and-ancova-in-a-designed-experiment](https://stats.stackexchange.com/questions/24077/how-to-choose-between-anova-and-ancova-in-a-designed-experiment)