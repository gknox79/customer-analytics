# Customer Analytics

This repo hosts the course content of Customer Analytics, taught at Tilburg University by George Knox.  This meant as a resource for teachers and students.  Corrections and comments are always welcome.  Excellent course assistance provided by Gijs van Bussel & Anne van der Vliet.  All errors are mine.  

## Course Description

Customer analytics means combining statistical models to predict customer behavior with simple economic models to derive optimal managerial policies. We look at predicting behavior from the level of the individual customer, as in true one-to-one marketing, scaled up to a firm's entire customer base. Companies have more data about their customers than ever before. But can they use this data to increase customer loyalty, prevent attrition, target marketing to customer interest, or calculate each customer’s worth to the organization?  It’s hard to act intelligently on such information, as they may be “drowning in data” yet “starving for insights” because they have no way to organize their data within a larger statistical framework. 

In this course, students will learn the modeling-based approach to marketing with the use of data sets, models, and using the results. We take a hands-on approach with databases and working in R in order to equip students with tools that can be used immediately on the job.

We will cover three broad areas: (1) test and roll (1 session), (2) next-period response models (4 sessions), and (3) customer lifetime value (CLV) models (2 sessions). Testing is important to determine what marketing works best before rolling out to larger groups; we talk about the bootstrap to quantify uncertainty. Response models are good for predicting what happens next period, e.g., immediately following some action by the firm (e.g., recommending a new product). CLV is about forecasting behavior over a longer horizon, the customer lifecycle.

We will examine a few different contexts using models to analyze customer behavior, including:

* **Test and Roll:** When should a test be conducted, how large should the test be?  How uncertain are we about an estimate response?
* **Next-period response models:** Which customers should be selected for e.g., acquisition, retention, cross-selling, up-selling?
  - RFM analysis
  - Logistic regression
  - LASSO
  - Decision trees
  - Random Forests
* **Collaborative filtering, next-product-to-buy models:** Which products should be recommended to which customers?
* **Long term prediction models for Customer Lifetime value:** How different are customers? How does their behavior change over time? What's the value of the customer over his or her lifecycle to the firm?  How does that change as a function of recency and frequency?  
  - Contractual settings
  - Non-contractual settings
 

## Course Purpose:

Managers (as well as consultants, analysts, and investors) are increasingly tasked with providing valid answers to such questions. Yet many of them lack the tools to address these problems. This course is designed to give you the powerful, cutting-edge tools to address these issues.

The course is organized around several case studies that illustrate an important concept with data. All these examples will be “hands-on” and have an emphasis on real-time problem-solving. You will develop the necessary skills to estimate these models and evaluate their results in R. 

## Learning Goals

* For a given model, students should be able to interpret results and explain intuitively what it assumes about customer behavior.
* Students should understand how to validate models and avoid overfitting.
* Students should be able to fit these models to real-world data; they should also be able to evaluate predictions across models and derive the managerial implications.
* Students should be able to calculate customer lifetime value (CLV).
Understand the role of unobserved heterogeneity in measuring customer loyalty.
