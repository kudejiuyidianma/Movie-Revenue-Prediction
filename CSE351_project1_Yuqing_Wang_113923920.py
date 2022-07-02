#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# # Data Exploration

# In[2]:


# load two csv files
credit = pd.read_csv('./movie/tmdb_5000_credits.csv')
movie = pd.read_csv('./movie/tmdb_5000_movies.csv')


# In[3]:


credit.info()


# In[4]:


credit.head(5)


# From the first 5 rows of credit table, we found that some variables are lists. We need to extract useful variables from those lists. 

# In[5]:


movie.info()


# In[6]:


movie.head(5)


# Observing 5 rows of movie table, we need to split genres and drop irrelevant columns.

# # Data Cleaning
# ## merge two table on movie id

# In[7]:


movie = movie.rename(columns={"id":"movie_id"})


# In[8]:


# merge two dataset with their common column
raw_df = pd.merge(movie, credit, on = 'movie_id')
raw_df.head()


# In[9]:


raw_df.info()


# In[10]:


movie.shape, credit.shape, raw_df.shape


# From the statistics information above, we have 4803 observations of 23 variables. We also acquire the data type of each factor.

# ## Handle Json Columns
# There are 6 variables which are list type. We use convert function to split and extract useful information from them.

# In[11]:


json_cols = ['genres', 'keywords', 'production_companies','production_countries',
            'spoken_languages','cast', 'crew']
raw_df[json_cols].head()


# In[12]:


import ast


# In[13]:


raw_df.genres[0]


# In[14]:


def convert(data):
    result = []
    for i in ast.literal_eval(data):
        result.append(i['name'])
    return result


# In[15]:


raw_df['genres'] = raw_df['genres'].apply(convert)
raw_df['genres'].head()


# In[16]:


raw_df['keywords'] = raw_df['keywords'].apply(convert)
raw_df['production_companies'] = raw_df['production_companies'].apply(convert)
raw_df['production_countries'] = raw_df['production_countries'].apply(convert)


# ## Preserve top 4 cast and the director from the cast list

# In[17]:


# extract top 4 cast
def top4_cast(data):
    cnt = 0
    result = []
    for i in ast.literal_eval(data ):
        if cnt<4:
            result.append(i['name'])
            cnt = cnt+1
        if cnt>4:
            break
        
    return result


# In[18]:


raw_df['cast'] = raw_df['cast'].apply(top4_cast )


# In[19]:


raw_df['cast'][0]


# In[20]:


# extract director from the cast
def director(data):
    for i in ast.literal_eval(data):
        if i['job'] == 'Director':
            return i['name']


# In[21]:


raw_df['Director'] = raw_df['crew'].apply(director)
raw_df['Director'].head()


# ## Handle Missing Values

# In[22]:


raw_df.isnull().sum().sort_values(ascending=False)


# The table above shows the missing values in the table. Homepage, tagline, and overview are not helpful for our data analysis. So we dropped them. title_x and title_y are duplicated by selecting one of them and removing the other one. The unknown release_date is made today and the missing value of runtime is filled with the median.

# In[23]:


raw_df.drop(['homepage' , 'tagline', 'overview','title_x'] , axis = 1 , inplace = True)
raw_df = raw_df.rename(columns={'title_y':'title'})


# In[24]:


raw_df['release_date'].fillna(datetime.today().strftime('%m/%d/%Y'), inplace=True)


# In[25]:


raw_df['runtime'].fillna(raw_df['runtime'].median(), inplace=True)


# ## Handle Outliers
# In this dataset, revenue is the target variable and budget is an important factor. So we draw distribution of the revenue and budget  to check if there are outliers.

# In[26]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['revenue'])
plt.title('Distribution of Revenue before Dropping Outliers')
plt.show()


# In[27]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['budget'])
plt.title('Distribution of Budget before Dropping Outliers')
plt.show()


# In[28]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['vote_count'])
plt.title('Distribution of Vote_count before Dropping Outliers')
plt.show()


# Two figures above show that most of revenue and budget of movies are under 500,000,000 and 100,000,000. However there are many outliers need to be dropped. So, we determined that the value higher than 95% and lower than 5% are outliers.

# In[29]:


def outlier(df, cname):
    maxthresold = df[cname].quantile(0.95)
    minthresold = df[cname].quantile(0.05)
    df = df[(df[cname]<maxthresold) & (df[cname]>minthresold)]
    return df


# In[30]:


for i in ['budget', 'revenue', 'vote_count']:
    raw_df = outlier(raw_df, i)


# In[31]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['revenue'])
plt.title('Distribution of Revenue after Dropping Outliers')
plt.show()


# In[32]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['budget'])
plt.title('Distribution of Budget after Dropping Outliers')
plt.show()


# In[33]:


plt.figure(figsize=(10,8))
sns.boxplot(y=raw_df['vote_count'])
plt.title('Distribution of vote_count after Dropping Outliers')
plt.show()


# After dropping outliers, there only have less outlier which are acceptable. Make a copy of our cleaned data.

# In[34]:


data = raw_df.copy()


# Now, we get the cleaned data.

# # Data Visualization
# ## Release Date
# We want to split release date into its year, month and day. Such that, we could count the number of movies released by day of week, month and year.

# In[35]:


data['release_date'].head()


# In[36]:


data['release_date'] = pd.to_datetime(data['release_date'])
lst = ['year','month','weekday']
for i in lst:
    data[i] = getattr(data['release_date'].dt, i).astype('int')
data.head()


# ### Movie Release Count per Week

# In[37]:


plt.figure(figsize = (10,8))
sns.countplot(x='weekday', data = data)
plt.title("Movie Release Count per Week")
plt.show()


# From the distribution of the movies release per week, we can find the most films released on Friday. This is followed by Thursdays and Wednesdays. Saturday and Sunday, on the contrary, have the least released movies.

# In[38]:


plt.figure(figsize = (10,8))
sns.countplot(x='month', data = data)
plt.title("Movie Release Count per Month")
plt.show()


# From the monthly movie release distribution chart, we can see that the most movies were released in September while the least movies were released in May.

# In[39]:


plt.figure(figsize = (20,16))
sns.countplot(x='year', data = data)
plt.title("Movie Release Count per Year")
plt.xticks(rotation = 'vertical')
plt.show()


# The annual distribution of movie releases illustrates that the earliest movie release recorded in the dataset was in 1916. There have been a small number of movie releases since then. It was not until after 1980 that the film industry came to a booming period. The number of movie releases increased dramatically. In the 21st century, the number of movie releases exploded even more, with as many as 150 movies released worldwide in a year.

# ### Correlation Heatmap

# In[40]:


pearsoncorr = data.loc[:,['revenue','popularity','runtime','vote_average','vote_count','budget']].corr(method='pearson')
sns.heatmap( pearsoncorr, 
            xticklabels=pearsoncorr.columns, 
            yticklabels=pearsoncorr.columns, 
            cmap='RdBu_r', 
            annot=True, 
            linewidth=0.5 )


# From the heatmap of the Pearson correlation, we could find that the budget has the most positive correlation. It indicates that the higher budget the movie input, the higher revenue the movie will get. Vote average has the least positive correlation with the revenue. 

# ### Genre Trend Shifting Patterns

# In[41]:


genres_df = data['genres'].apply(pd.Series)
genres_df.head()


# In[42]:


stacked_genres = genres_df.stack()
stacked_genres.head()


# In[43]:


raw_dummies = pd.get_dummies(stacked_genres)
raw_dummies.head()


# In[44]:


genres_dummies = raw_dummies.sum(level=0)
genres_dummies.head(3)


# In[45]:


genres_dummies['year'] = data['year']
genres_dummies.head(3)


# In[46]:


grouped = genres_dummies.groupby('year')
groupedCnt = grouped.agg(np.sum).transpose()
groupedCnt


# In[47]:


plt.figure(figsize=(27,11))
cmap = sns.cubehelix_palette(start=1.5, rot=1.5, as_cmap=True)
sns.heatmap(groupedCnt, xticklabels=3, cmap=cmap, linewidths=0.05).invert_yaxis()
plt.title('Movie Genre Trend Heatmap')
plt.show()


# In[48]:


groupedCnt.T.sort_index()[:-1].plot.line(figsize = (20,10))


# From the heatmap, we could easily find the five most popular generes which are 'Drama', 'Comedy', 'Thriller', 'Action' and 'Romance'. From the line graph we can see that ‘Drama’ has been a popular category since movies first appeared and all types of movies have shown a growth trend.

# # Modeling
# ## Choose Features
# According to the correlation matrix above, we found that budget, popularity and vote_count have a higher correlation with the revenue. However, the correlation of vote_count and popularity is too high. Hence, we only choose vote_count and budget in the following models.

# In[49]:


data[['revenue', 'budget', 'vote_count']].describe()


# In[82]:


sns.pairplot(data, x_vars=['vote_count','budget'], y_vars='revenue',kind="reg", height=5, aspect=0.8)
plt.show()


# ## Feature Scaling
# The description of the data shows enormous gaps between the largest/smallest and median values. It means no coefficient can use the feature without blowing up on big values.
# Thus, we replace such features x with log(x).

# In[50]:


data["logre"] = data['revenue'].map(lambda x:np.log(x+1))
data["logbud"] = data['budget'].map(lambda x:np.log(x+1))
data["logvote"]=data['vote_count'].map(lambda x:np.log(x+1))


# ## Implement Models

# In[51]:


import math
def RSE(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


# ### Linear Regression

# After data processing, we made prediction of revenue based on vote_count and budget multi-linear regression. In the graph on the right, the X-axis represents vote_count and budget, and the Y-axis represents revenue. According to the previous Heatmap of the Pearson correlation, it can be seen that popularity, vote_count and budget are most correlated with revenue. I chose vote_count and budget because popularity and Vote_count have a strong correlation of 0.87. If I put three variables there, then the collinear phenomena may occur. Since the variables of the X-axis are highly correlated, they will affect the prediction, and other variables cannot be fixed, so the real relationship between X and Y cannot be found.

# In[83]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
dfm1 = data[["logre","logbud","logvote"]]


# In[84]:


X = dfm1[["logbud","logvote"]].values
y = dfm1.logre.values


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
md1 = LinearRegression()  
md1.fit(X_train, y_train)


# In[86]:


y_pred = md1.predict(X_test)


# In[87]:


acc = y_test
pre = y_pred
plt.figure(figsize = (10,8))
plt.plot(acc,label = 'real values')
plt.plot(pre, label = 'prediction')
plt.legend()
plt.title('Prediction Vs. Real values')
plt.show()


# In[88]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


# When making predictions, the mean absolute error between the predicted and observed values is 0.78, I think it's good, because it represents the absolute difference between the actual value and the forecast value.
# 

# Mean Squared error is 1.35, there is no overfit in this group of predictions. Because MSE equals 0 is theoretically the best, the closer you get to 0, the better prediction you get. However, if the MSE is too small, it may indicate that the model is overfit; if the MSE is too large, it may indicate that the model is underfit.

# The root mean squared error is 1.16. This means it fits the data fairly well, because it's close to zero.

# In[89]:


from sklearn.model_selection import cross_val_score
scores = -cross_val_score(md1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(scores)
print(scores.mean())


# We separate the training data into 5 groups to make cross-validation and this is the mean value of five scores. This process is used to estimate the skill of the model on new data and overcome the overfitting and underfitting problems. Now, the cross validation score shows a good performance of the multiple linear regression model in other data sets not only in the train and test data set. But, we still need to compare this model with the rest two models.

# In[90]:


print(RSE(y_test, y_pred))


# This is the residual standard error of the multi-linear regression. It is close to 0 which means the model is accuracy.

# ## Logistic Regression

# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dfm2 = data[["logre","logbud","logvote"]]

for index, row in dfm2.iterrows():
    if row["logbud"] >= row["logre"]:
        dfm2.loc[index, "Target"] = 0
    else:
        dfm2.loc[index, "Target"] = 1
dfm2.Target = dfm2.Target.astype(int)


# In[92]:


X = dfm2[["logbud", "logvote"]].values
y = dfm2.Target


# In[93]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[94]:


md2 = LogisticRegression()
md2.fit(x_train, y_train)


# In[95]:


predictions = md2.predict(x_test)


# In[96]:


print(RSE(y_test, predictions))


# This is the residual standard error of the logistic regression. It shows a larger value than the residual standard error of the multiple linear regression. Hence, multiple linear regression is better than logistic regression in the test data set.
# 

# In[97]:


scores = cross_val_score(md2, X_train, y_train, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())


# The mean value of the cross validation score is 0.76. It shows a decrease in the score compared to the mean score of multiple linear regression cross validation. This indicates that the logistic regression model shows a lower accuracy.

# ## Polynomial Regression
# Polynomial Regression is a form of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. 

# In[98]:


train_df = data[:int(len(data)*0.8)]
test_df = data[int(len(data)*0.8):]


# In[99]:


train_x = train_df[['logvote']].values
train_y = train_df['revenue'].apply(np.log1p).values
test_x = test_df['logvote'].values
test_y = test_df['revenue'].apply(np.log1p).values


# In[100]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
mse = []
m = 1
m_max = 10
train_x = train_x.reshape(len(train_x),1)
test_x = test_x.reshape(len(test_x),1)
train_y = train_y.reshape(len(train_y),1)
while m<=m_max:
    model = make_pipeline(PolynomialFeatures(m, include_bias=False),LinearRegression())
    model.fit(train_x,train_y)
    pre_y = model.predict(test_x)
    mse.append(mean_squared_error(test_y, pre_y.flatten()))
    m = m+1
plt.figure(figsize=(10,6))
plt.plot([i for i in range(1, m_max+1)], mse, 'r')
plt.scatter([i for i in range(1, m_max+1)], mse)
plt.title('MSE of m degree of polynomial regression')
plt.xlabel('m')
plt.ylabel('MSE')
plt.show()


# In Polynomial Regression, I first set degree of polynomial regression from 1 to 10. Then, I record the mean square error for each time. From the graph above, we can see that when the degree equals to 5 the mean square error be the least. So, we decided the degree of polynomial regression is 5.

# In[101]:


model = make_pipeline(PolynomialFeatures(5, include_bias=False),LinearRegression())
model.fit(train_x,train_y)
pre_y = model.predict(test_x)
acc = test_y
pre = pre_y
plt.figure(figsize = (20,10))
plt.plot(acc,label = 'real values')
plt.plot(pre, label = 'prediction')
plt.xlabel('log vote count')
plt.ylabel('log revenue')
plt.legend()
plt.title('Prediction Vs. Real values')
plt.show()


# In[102]:


print(RSE(test_y, pre_y))


# This is the residual standard error of the polynomial regression. We could easily find that this number is much larger than the residual standard error of the multiple linear regression. Hence, the multiple linear regression model is much better than the polynomial regression model in the test data set.

# In[103]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("polynomial regression mean absolute error: ", mean_absolute_error(test_y,pre_y.flatten()))
print("polynomial regression mean squared error: ", mean_squared_error(test_y,pre_y.flatten()))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


# The mean absolute error between the predicted and observed values is 1.63. It is higher than the mean absolute error than multiple linear regression. It indicates the error between true value and predict value is larger.

# Mean Squared error is 5.68 which indicates that this model is not fitter than the multiple linear regression.

# This is the root mean square error of polynomial regression. It indicates that the model do not fit the data well since the value of RMSE is too large.

# In[104]:


from sklearn.model_selection import cross_val_score
scores = -cross_val_score(model, train_x, train_y, cv=5, scoring='neg_mean_absolute_error')
print(scores)
print(scores.mean())


# This is the mean value of the five cross validation scores. This value is larger than the value of logistic regression but lower than the multiple linear regression. Only discussing the cross validation score, the polynomial regression model shows a better performance in new datasets than the logistic regression.

# # Conclusion
# From the exploration data analysis, we could conclude that more movie producers prefer release movies on Fridays or summer. 
# 
# It was not until after 1980 that the film industry came to a booming period. 
# 
#  ‘Drama’ has been a popular category since movies first appeared and all types of movies have shown a growth trend from about 1990. Between 2005 and 2010, most genre films reached their peak. And after 2010, there was a certain degree of decline.
#  
# Vote_count is the most correlated factor in predicting the revenue of the movie, while vote average is the least correlated factor.
# 
# 
# According to the residual standard error and the mean cross validation score, the multiple linear regression gives the best performance compared to the rest two models.
# 
# It has the least residual standard error which indicates that it is the most accuracy in the test data set. Meanwhile, it has the highest mean cross validation score which indicates that it also have a good performance in the new datasets rather than the train and test dataset.
# 
# Besides, we suppose that the polynomial regression is better than the logistic regression. Firstly, the polynomial regression is a numeric prediction model while the logistic regression is a classification prediction model. Then, the logistic regression has the worst performance in new datasets. It will give inaccuracy prediction if we use this model to make a prediction of the movie revenue.
