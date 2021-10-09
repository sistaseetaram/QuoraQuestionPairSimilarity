## Business Problem
### Description
Ref: https://www.kaggle.com/c/quora-question-pairs/ 

* Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.<br/> <br/>

* Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

### Problem Statement
* Identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered. <br/>
** We are tasked with predicting whether a pair of questions are duplicates or not.**

### Data overview
* Data will be in a file Train.csv <br>
* Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
* Size of Train.csv - 60MB <br>
* Number of rows in Train.csv = 404,290

### Performance Metric (decided by kaggle)
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
1. log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
2. Binary Confusion Matrix
