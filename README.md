# Customer Analytics

This repo hosts the course content of Customer Analytics, a 7-week course taught at Tilburg University by George Knox.  This is meant as a resource for teachers and students in the spirit of open education.  Corrections and comments are always welcome.  Excellent course assistance provided by Gijs van Bussel & Anne van der Vliet.  All errors are mine.  

## Course Description

Customer analytics means combining statistical models to predict customer behavior with simple economic models to derive optimal managerial policies. We look at predicting behavior from the level of the individual customer, as in true one-to-one marketing, scaled up to a firm's entire customer base. Companies have more data about their customers than ever before. But can they use this data to increase customer loyalty, prevent attrition, target marketing to customer interest, or calculate each customer’s worth to the organization?  It’s hard to act intelligently on such information, as they may be “drowning in data” yet “starving for insights” because they have no way to organize their data within a larger statistical framework. 

In this course, students will learn the modeling-based approach to marketing with the use of data sets, models, and using the results. We take a hands-on approach with databases and working in R in order to equip students with tools that can be used immediately on the job.

We will cover three broad areas: (1) test and roll (1 session), (2) next-period response models (4 sessions), and (3) customer lifetime value (CLV) models (2 sessions). Testing is important to determine what marketing works best before rolling out to larger groups; we talk about the bootstrap to quantify uncertainty. Response models are good for predicting what happens next period, e.g., immediately following some action by the firm (e.g., recommending a new product). CLV is about forecasting behavior over a longer horizon, the customer lifecycle.

We will examine a few different contexts using models to analyze customer behavior, including:

* **Test and Roll:** When should a test be conducted, how large should the test be?  How uncertain are we about an estimate response?
* **Next-period response models:** Which customers should be selected for e.g., acquisition, retention, cross-selling, up-selling?
  - RFM analysis
  - Logistic regression
  - Subset selection
  - LASSO
  - Decision trees
  - Random Forests
* **Collaborative filtering, next-product-to-buy models:** Which products should be recommended to which customers?
* **Long term prediction models for Customer Lifetime value:** How different are customers? How does their behavior change over time? What's the value of the customer over his or her lifecycle to the firm?  How does that change as a function of recency and frequency?  
  - Contractual settings (sBG)
  - Non-contractual settings (BG/BB, Pareto/NBD and BG/NBD)
 

## Course Purpose:

Managers (as well as consultants, analysts, and investors) are increasingly tasked with providing valid answers to such questions. Yet many of them lack the tools to address these problems. This course is designed to give you the powerful, cutting-edge tools to address these issues.

The course is organized around several case studies that illustrate an important concept with data. All these examples will be “hands-on” and have an emphasis on real-time problem-solving. You will develop the necessary skills to estimate these models and evaluate their results in R. 

## Learning Goals

* For a given model, students should be able to interpret results and explain intuitively what it assumes about customer behavior.
* Students should understand how to validate models and avoid overfitting.
* Students should be able to fit these models to real-world data; they should also be able to evaluate predictions across models and derive the managerial implications.
* Students should be able to calculate customer lifetime value (CLV).
Understand the role of unobserved heterogeneity in measuring customer loyalty.

## Syllabus

### Week 1: Test & Roll, Uncertainty

* Test & Roll
* Option Value of Testing
* Classical Uncertainty
* Bayesian Uncertainty
* Comparing Posteriors
* Size of Test Group

[Database Marketing Blattberg Kim and Neslin (2008) henceforth BKN Sections 9.1-2](http://link.springer.com/book/10.1007%2F978-0-387-72579-6)
[Elea McDonnell Feit, Ron Berman (2019) Test & Roll: Profit-Maximizing A/B Tests. Marketing Science 38(6):1038-1058.](https://
doi.org/10.1287/mksc.2019.1194)

### Week 2: Next period models: RFM

* RFM Analysis
* Empirical Bayes
* Clumpiness

[Quick Profits with RFM Analysis](http://www.dbmarketing.com/articles/Art149.htm)
[BKN Chapter 12: RFM Analysis](http://link.springer.com/book/10.1007%2F978-0-387-72579-6)

### Week 3: Logistic Regression, Overfitting and Optimal Targeting

* Customer Churn
* Logistic regression
* Overfitting
* Lifts and Optimal Targeting

[BKN: Chapter 15 (377-85), Chapter 10, and 11.4.2](http://link.springer.com/book/10.1007%2F978-0-387-72579-6)
Ascarza, Eva. "Retention futility: Targeting high-risk customers might be ineffective." Journal of Marketing Research 55.1 (2018): 80-98.

### Week 4: Other next period models

* Subset Selection
* LASSO
* Decision Trees
* Random Forests


[BKN Chapter 17, 11.4](https://link.springer.com/book/10.1007%2F978-0-387-72579-6)
[James, Gareth, et al. An introduction to statistical learning. Vol. 112. New York: springer, 2013. Sections: 6.1-2, 8.1-2](https://web.stanford.edu/~hastie/ISLR2/ISLRv2_website.pdf)
[Varian, Hal R. "Big data: New tricks for econometrics." Journal of Economic Perspectives 28.2 (2014): 3-28.](https://pubs.aeaweb.org/doi/pdf/10.1257/jep.28.2.3)

### Week 5: Customization

* Next product to buy models
* Recommender Systems

[Recommendation Systems](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)
[BKN Chapter 21: Cross-Selling and Up-Selling](https://link.springer.com/chapter/10.1007/978-0-387-72579-6_21)
[Knott, Aaron, Andrew Hayes, and Scott A. Neslin. "Next‐product‐to‐buy models for cross‐selling applications." Journal of interactive Marketing 16.3 (2002): 59-75.](https://doi.org/10.1002/dir.10038)
[Winning the Netflix Prize](https://youtu.be/ImpV70uLxyw)

### Week 6: Customer base analysis: Contractual Settings

* CLV definition
* Geometric model
* RLV and CLV
* Heterogeneity and retention rates
* shifted Beta-geometric model

[Fader, Peter S., and Bruce GS Hardie. "How to project customer retention." Journal of Interactive Marketing 21.1 (2007): 76-90.](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf)
[Fader, Peter S., and Bruce GS Hardie. "Customer-base valuation in a contractual setting: The perils of ignoring heterogeneity." Marketing Science 29.1 (2010): 85-93.](https://doi.org/10.1287/mksc.1080.0482)
[A Spreadsheet-Literate Non-Statistician’s Guide to the Beta-Geometric Model](http://www.brucehardie.com/notes/032/)


### Week 7: Customer base analysis: Non-contractual Settings

* Noncontractual vs. Contractual
* Beta-geometric Beta-Binomial model
* CLV & RLV
* Extensions: Pareto/NBD, BG/NBD


[Fader, Peter S., Bruce GS Hardie, and Jen Shang. "Customer-base analysis in a discrete-time noncontractual setting." Marketing Science 29.6 (2010): 1086-1108.](https://doi.org/10.1287/mksc.1100.0580)
[Fader, Peter S., Bruce GS Hardie, and Ka Lok Lee. "“Counting your customers” the easy way: An alternative to the Pareto/NBD model." Marketing science 24.2 (2005): 275-284.](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)



### Week 8: Wrap up

