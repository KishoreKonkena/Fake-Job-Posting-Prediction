# Fake-Job-Posting-Prediction

Project Overview
Employment scams are on the rise. According to CNBC, the number of employment scams doubled in 2025 as compared to 2023 and 2024. The current market situation has led to high unemployment. Economic stress and the impact of the coronavirus have significantly reduced job availability and the loss of jobs for many individuals. A case like this presents an appropriate opportunity for scammers. Many people are falling prey to these scammers using the desperation that is caused by an unprecedented incident. Most scammer do this to get personal information from the person they are scamming. Personal information can contain address, bank account details, social security number etc. I am a university student, and I have received several such scam emails. The scammers provide users with a very lucrative job opportunity and later ask for money in return. Or they require investment from the job seeker with the promise of a job. This is a dangerous problem that can be addressed through Machine Learning techniques and Natural Language Processing (NLP).

This project uses data provided from Kaggle. This data contains features that define a job posting. These job postings are categorized as either real or fake. Fake job postings are a very small fraction of this dataset. That is as excepted. We do not expect a lot of fake jobs postings. This project follows five stages. The five stages adopted for this project are –

Problem Definition (Project Overview, Project statement and Metrics)

Data Collection

Data cleaning, exploring and pre-processing

Modeling

Evaluating

**Problem Statement**

This project aims to create a classifier that will have the capability to identify fake and real jobs. The final result will be evaluated based on two different models. Since the data provided has both numeric and text features one model will be used on the text data and the other on numeric data. The final output will be a combination of the two. The final model will take in any relevant job posting data and produce a final result determining whether the job is real or not.


📊 Dataset

Source: [Fake Job Posting Dataset (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

Contains ~18,000 job postings with features like:

title, location, department, salary_range, company_profile, description, requirements, benefits

Binary target column: fraudulent (0 = Real, 1 = Fake)


**Metrics**

The models will be evaluated based on two metrics:

1. Accuracy: This metric is defined by this formula -
2.  

As the formula suggests, this metric produces a ratio of all correctly categorized data points to all data points. This is particularly useful since we are trying to identify both real and fake jobs unlike a scenario where only one category is important. There is however one drawback to this metric. Machine learning algorithms tend to favor dominant classes. Since our classes are highly unbalanced a high accuracy would only be a representative of how well our model is categorizing the negative class (real jobs).

F1-Score: F1 score is a measure of a model’s accuracy on a dataset. The formula for this metric is –
equation

F1-score is used because in this scenario both false negatives and false positives are crucial. This model needs to identify both categories with the highest possible score since both have high costs associated to it.
