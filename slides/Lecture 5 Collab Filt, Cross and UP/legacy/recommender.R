
install.packages("recommenderlab")
library("recommenderlab")	 	 

# Loading to pre-computed affinity data	 

# load movie rating data from website MovieLense
data("MovieLense")

## matrix of ratings dimensions (how many users rating how many movies)
MovieLense100 <- MovieLense[rowCounts(MovieLense) >100,]

as(MovieLense100@data@Dimnames, "list")[[2]][1:10]

## look at the first few ratings of the first user
head(as(MovieLense100[1,], "list")[[1]])

## visualize part of the matrix
image(MovieLense100)

## number of ratings per user
hist(rowCounts(MovieLense), breaks=50)

## number of ratings per movie
hist(colCounts(MovieLense), breaks=50)

# distribution of ratings across movies
hist(getRatings(MovieLense))

## mean rating (averaged over users)
mean(rowMeans(MovieLense))


## 
train <- MovieLense100[1:50]
rec <- Recommender(train, method = "UBCF")
rec

rec_pop<-Recommender(MovieLense[1:942],method="POPULAR")

rec_pop

names(getModel(rec_pop))

getModel(rec_pop)$topN

predicted_ratings<-predict(rec_pop, MovieLense[943],n=5)

predicted_ratings


as(predicted_ratings,"list")

recom <- predict(rec_pop, MovieLense[943], type="ratings")

recom

as(recom)


m<-as(MovieLense,"realRatingMatrix")

as()


getRatingMatrix(m)

UCF<-recommender(m[1:5000],method="UBCF")

movies <- as(MovieLense, 'data.frame')
movies$user <- as.numeric(movies$user)
movies$item <- as.numeric(movies$item)

sparse_ratings <- sparseMatrix(i = movies$user, j = movies$item, x = movies$rating, 
                               dims = c(length(unique(movies$user)), length(unique(movies$item))),  
                               dimnames = list(paste("u", 1:length(unique(movies$user)), sep = ""), 
                                               paste("m", 1:length(unique(movies$item)), sep = "")))

real_ratings <- new("realRatingMatrix", data = sparse_ratings)
real_ratings
























## Real data
```{r message=FALSE}
rm(list=ls())

#install.packages("recommenderlab")
library("recommenderlab")	

data("MovieLense")


```

### Recommender System, Matrix factorization
This notebook constructs a simple version of a more advanced recommender system on a subset of a subset of the movielense dataset. The recommender system relies on matrix factorization to extract latent product and user features. Thereafter it uses these features to make predictions and generate recommendations. 

### Loading data and creating a subset
The following code chunk loads packages for data-manipulation. Thereafter it loads in the data of the MovieLense dataset, where each line represents the rating of a user on a movie, identified and structured as: UserId, MovieId, Rating and TimeStamp. We decrease the number of movies in the dataframe (to make the workload manageable on your PC), and delete users who do not have more than five interactions. The timestamp column is also deleted, since we do not use it. 

library(readxl)
library(dplyr)
library(reshape2)
library(recosystem)
library(Metrics)

df <- read.csv('data\\ratings.csv',sep = ',', header = TRUE, stringsAsFactors = FALSE) #Dataframe containing the ratings
names <- read.csv('data\\movies.csv',sep = ',', header = TRUE, stringsAsFactors = FALSE) #Dataframe containing the names and genres of movies
df <- df[,-4] #Delete Timestamp
#length(unique(df$movieId)) #Check number of unique movieId's in the sample

#df <- df[!(df$movieId >1000),] #Decrease movie sample, only movieId's below 1000 are now in the sample

length(unique(df$movieId)) # There are now 761 movieId's left in the sample
#length(unique(df$userId)) # We originally have 603 users
#df <- df %>% group_by(userId) %>% filter(n()>5) # we clean the matrix for users who have only seen less than 6 movies
length(unique(df$userId)) # We now have 508 users left

```

# Creating new Movie- and UserIds
Given operations later in this notebook, it is handy to replace the original movie- and userId by enumerating the current Id's. It can be the case that there are jumps within these id's: for instance, we might go from movieId 83 to movieId 98, with no movieId's in between. Later on it is necessary, to have a steadily increasing number, particullarly, the movieId jumps from 83 to 84 with an increase of 1, instead of jumping 15 numbers forward. Same goes for userId's, which were originally steadily increasing, but by deleting user with less than 6 ratings, we've undone that consistency.  

```{r}
#df1<-df #Create new dataframe, which contains all old and new_movieId's
nrow(df)
df$new_movie_ID <- df %>% #Creating new movieId: Group by movieId, placing indices and storing them in a new variable
  group_by(movieId) %>%
  group_indices(.) 
df$new_user_ID <- df %>% #Creating new userId: Group by userId, placing indices and storing them in a new variable
  group_by(userId) %>%
  group_indices(.)
nrow(df)
```

# Splitting the dataset
In the following code-chunk we create a validation set for the performance of our recommender system. We group all the ratings of a user, whereafter we sample a single row. Then we join the movies with their *new_movie_ID* and *new_user_ID*. Thus we will validate our data by leaving one of each user's original ratings out of the training data for the recommender system, and then compare the recommender system's prediction to the actual value. 

```{r}
set.seed(8) #Making sure that the randomness is always the same (but it is still random)
test_df <- df %>% group_by(userId) %>% sample_n(1) #sample a observation per user that we're going to delete in the original DF
#These are the values that we're going to predict, thus we store them in a test set
test_df #The validation set
```

We delete the observations that are in the test set from the original training set. 
```{r}
nrow(df) #Total amount of ratings
train_df_long <- anti_join(df, test_df, by=c('new_user_ID', 'new_movie_ID', 'rating') ) #delete the observations in the original dataset that are in the test set
nrow(train_df_long) #Total amount of ratings after deletion
```

```{r}
names <- unique(right_join(names, df[,c("new_movie_ID","movieId")], by=c("movieId","movieId"))) #join the new_movie_ID to the names dataframe
```

# Constructing the matrix
```{r}
train_m_wide <- acast(train_df_long, new_user_ID~new_movie_ID, value.var="rating")
train_m_wide[is.na(train_m_wide)] = 0 
dim(train_m_wide)
```

```{r}
#Write the dataset in a csv localy, needed for the Reco() object
write.table(train_df_long, file="train.csv", row.names = F, col.names = F)
write.table(test_df, file="test.csv", row.names = F, col.names = F)
```


Credits for this code go to https://github.com/yixuan/recosystem

The following picture shows the essence of matrix factorization.^[Integrating spatial and temporal contexts into a factorization model for POI recommendation - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Diagram-of-matrix-factorization_fig1_321344494 [accessed 6 Jul, 2020]]  
![Figure 1. Diagram of matrix factorization.](Diagram-of-matrix-factorization.png) 

By decomposing the original the rating of the user-item matrix (in our code, that is the variable 'train_df_long') into two lower dimensional rectangular matrices, the method can infer predictions on unknown user-item interactions, or movie-ratings in our case. Matrix factorization creates the two laten matrices, and by minimizing a cost function based on the original known interactions, i.e. $\min_{p_u, q_i}\sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2$, it is able to generate predictions for unknown interactions. 

It is optimized with stochastic gradient descent. The method uses the derivative of the cost function with respect to the user latent factors, i.e. $$\frac{\partial}{\partial p_u}(r_{ui} - p_u \cdot q_i)^2= - 2 q_i (r_{ui} - p_u \cdot q_i) $$, moves a step closer to the lowest possible value, and then applies the same trick with the item latent factors, i.e. $$ \frac{\partial}{\partial q_i}(r_{ui} - p_u \cdot q_i)^2 = - 2 p_u (r_{ui} - p_u \cdot q_i) $$ . By reapplying this trick, it hopes to find the global minimum of the cost function. As explained on [http://nicolas-hug.com/blog/matrix_facto_3], the basic version of the algorithm is as follows:
  
  1. Randomly initialize all vectors $p_u$ (latent user factors) and $q_i$ (latent user factors)
2. for a given number of times, repeat:
  * for all known ratings rui, repeat:
  + compute $\frac{\partial}{\partial p_u}$ and $\frac{\partial}{\partial q_i}  (r_{ui} - p_u \cdot q_i)^2$ 
  + update $p_u$ and $q_i$ with the following rule: $p_u \leftarrow p_u + \alpha \cdot q_i (r_{ui} - p_u \cdot q_i)$, and $q_i \leftarrow q_i + \alpha \cdot  p_u (r_{ui} - p_u \cdot q_i)$. We avoided the multiplicative constant 2 and merged it into the learning rate $\alpha$.


```{r}
recommender <- Reco()
recommender$train(data_file('train.csv'),opts = c(dim=20, costp_12=0.1, costq_12=0.1,
                                                  lrate=0.1, niter=100, nthread=6, verbose=F))
outfile = tempfile('RMSE test', tmpdir = getwd())
predicted <- recommender$predict(data_file('test.csv'), out_memory())
actual <- as.matrix(test_df[3])
rmse(actual=actual, predicted=predicted)
```

This was a very preliminary fit with just out of the box settings.



# What's happening underneath the hood
http://nicolas-hug.com/blog/matrix_facto_4
```{r eval=FALSE}
options(warn=-1)
train_df_long1 <- train_df_long
#train_df_long1$new_movie_ID <- train_df_long %>%
# group_indices(movieId) 
train_df_long2 <- train_df_long1
train_df_long2$movieId <- NULL
train_df_long2 <- train_df_long2[,c(1,3,2)]
SGD <- function(data) {
  n_factors <- 20 #number of factors
  alpha <- 0.01 #learning rate
  n_epochs <- 100 #number of iterations
  
  n_users <- length(unique(data$userId))
  n_items <- length(unique(data$new_movie_ID))
  
  #randomly initialize the user and item factors 
  p <- matrix(rnorm(n_users*n_factors,mean=0,sd=0.15), n_users, n_factors)
  q <- matrix(rnorm(n_items*n_factors,mean=0,sd=0.15), n_items, n_factors)
  suppressWarnings(for (z in 1:n_epochs){
    for (j in 1:n_users) {
      u <- data[j,1]
      i <- data[j,2]
      r_ui <- data[j,3]
      err <- r_ui -  p[u,]%*%q[i,]
      # update p_u and q_i
      p[u,] <- p[u,] + as.vector(alpha*err*q[i,])
      q[i,] <- q[i,] + as.vector(alpha*err*p[u,]) 
    }
    cat("iteration = ", z)
  })
  return (list('p'=p,'q'=q))
}
recommender <- SGD(train_df_long2)
```


```{r eval=FALSE}
predictions <- recommender$p%*%t(recommender$q)
long_predictions <- melt(predictions)
names(long_predictions) <- c('userId','new_movie_ID', 'pred')
long_predictions <- inner_join(x=long_predictions, y=train_df_long1[,c('movieId', 'new_movie_ID')], by='new_movie_ID')
pred_act <- unique(inner_join(x=test_df, y=long_predictions, by=c('userId', 'movieId')))
rmse(actual=as.matrix(pred_act$rating), predicted=as.matrix(pred_act$pred))
```


# Recommender System, Nearest Neighbors
This notebook constructs a simple recommender system on a subset of a subset of the movielense dataset. Jaccard distance and cosine similarity measures are implemented. 

## Loading data and creating a subset
The following code chunk loads packages for data-manipulation. Thereafter it loads in the data of the MovieLense dataset, where each line represents the rating of a user on a movie, identified and structured as: UserId, MovieId, Rating and TimeStamp. We decrease the number of movies in the dataframe (to make the workload manageable on your PC), and delete users who do not have more than five interactions. The timestamp column is also deleted, since we do not use it. 
```{r}
#library(readxl)
library(dplyr)
library(reshape2)
library(Metrics)
df <- read.csv('ml-latest-small\\ratings.csv',sep = ',', header = TRUE, stringsAsFactors = FALSE) #Dataframe containing the ratings
names <- read.csv('ml-latest-small\\movies.csv',sep = ',', header = TRUE, stringsAsFactors = FALSE) #Dataframe containing the names and genres of movies
df <- df[,-4] #Delete Timestamp
length(unique(df$movieId)) #Check number of unique movieId's in the sample
df <- df[!(df$movieId >1000),] #Decrease movie sample, only movieId's below 1000 are now in the sample
length(unique(df$movieId)) # There are now 761 movieId's left in the sample
length(unique(df$userId)) # We originally have 603 users
df <- df %>% group_by(userId) %>% filter(n()>5) # we clean the matrix for users who have only seen less than 6 movies
length(unique(df$userId)) # We now have 508 users left
```

# Creating new Movie- and UserIds
Given operations later in this notebook, it is handy to replace the original movie- and userId by enumerating the current Id's. It can be the case that there are jumps within these id's: for instance, we might go from movieId 83 to movieId 98, with no movieId's in between. Later on it is necessary, to have a steadily increasing number, particullarly, the movieId jumps from 83 to 84 with an increase of 1, instead of jumping 15 numbers forward. Same goes for userId's, which were originally steadily increasing, but by deleting user with less than 6 ratings, we've undone that consistency.  
```{r}
#df1<-df #Create new dataframe, which contains all old and new_movieId's
df$new_movie_ID <- df %>% #Creating new movieId: Group by movieId, placing indices and storing them in a new variable
  group_by(movieId) %>%
  group_indices(.) 
df$new_user_ID <- df %>% #Creating new userId: Group by userId, placing indices and storing them in a new variable
  group_by(userId) %>%
  group_indices(.)
nrow(df)
```


# Splitting the dataset
In the following code-chunk we create a validation set for the performance of our recommender system. We group all the ratings of a user, whereafter we sample a single row. Then we join the movies with their *new_movie_ID* and *new_user_ID*. Thus we will validate our data by leaving one of each user's original ratings out of the training data for the recommender system, and then compare the recommender system's prediction to the actual value. 

```{r}
set.seed(8) #Making sure that the randomness is always the same (but it is still random)
test_df <- df %>% group_by(userId) %>% sample_n(1) #sample a observation per user that we're going to delete in the original DF
#These are the values that we're going to predict, thus we store them in a test set
test_df #The validation set
```

We delete the observations that are in the test set from the original training set. 
```{r}
nrow(df) #Total amount of ratings
train_df_long <- anti_join(df, test_df, by=c('new_user_ID', 'new_movie_ID', 'rating') ) #delete the observations in the original dataset that are in the test set
nrow(train_df_long) #Total amount of ratings after deletion
```

```{r}
names <- unique(right_join(names, df[,c("new_movie_ID","movieId")], by=c("movieId","movieId"))) #join the new_movie_ID to the names dataframe
```

# Constructing the matrix

Below is the example from the chapter on recommender systems, page 334.

## Figure 9.4
```{r}
utility_matrix <- matrix(c(4,5,NA,NA,
                           NA,5,NA,3,
                           NA,4,NA,NA,
                           5,NA,2,NA,
                           1,NA,4,NA,
                           NA,NA,5,NA,
                           NA,NA,NA,3), nrow = 4, ncol = 7)
utility_matrix[is.na(utility_matrix)] = 0 
utility_matrix
```

We now construct the same matrix for the Movie-Dataset. However, we take the transpose of the matrix. Without diving to much into the details, we would like to find similar movies, which is usually more insightful for recommender systems than finding similar users. Taking the transpose of the matrix takes care of this particularity. 

## Movie Dataset
```{r}
train_m_wide <- t(acast(train_df_long, userId~movieId, value.var="rating"))
train_m_wide[is.na(train_m_wide)] = 0 
dim(train_m_wide) #We've computed a matrix of dimensions (761,508), or in other words, a utility matrix with 761 movies and 508 users 
#note that we have taken the transpose, so we have users in the columns and movies in the rows, as opposed to the example of Figure 9.4
```

# Getting Jaccard Distances
Jaccard Distances are calculated as:$\text{J}(\text{A,B})=\frac{\text{A}\cap\text{B}}{\text{A}\cup\text{B}}$^[https://en.wikipedia.org/wiki/Jaccard_index],
where $\text{A}\cap\text{B}$ is the number of overlapping rated movies by users A and B, and $\text{A}\cup\text{B}$ is the number of all rated movies by A and B. Thus we divide the union of the set of rated movies of two users by their entire combined set. 

## Example 9.7
```{r}
jaccard_dist_utility_m <- as.matrix(dist(utility_matrix, method = 
                                           'binary')) 
#the dist(, method='binary') function takes the jaccard distance
jaccard_dist_utility_m
```
Here we see that we find the same results as in the book's example. The distance between user A and B, in our case user 1 and 2, is 0.8, which means that the two users are quite distant. The users 3 and 1 are closer, with a Jaccard distance index of 0.5.

Now we apply the same calculation to the movie dataset. 

## Movie Dataset Jaccard Distances
```{r}
jaccard_dist <- as.matrix(dist(train_m_wide, method = 'binary'))
dim(jaccard_dist) #Check if we're doing item-to-item methods, dimensions should be overlapping with the number of unique movie-id's
```

# Getting Cosine Similarities

Now we turn to the second distance metric, cosine similarity. As opposed to the Jaccard distance above, a smaller value now represents two users or products being closer to one another. 
$\text{cosine  similarity} = \text{cos}(\theta) = \frac{A \cdot B}{||A||\cdot||B||} = \frac{\sum^n_{i=1}A_iB_i}{\sum^n_{i=1}A_i^2 \sum^n_{i=1}B_i^2}$

## Example 9.8
```{r}
cos_sim_utility_m <- utility_matrix / sqrt(rowSums(utility_matrix * utility_matrix))
cos_sim_utility_m <- cos_sim_utility_m %*% t(cos_sim_utility_m)
cos_sim_utility_m
```
Once again, we see that the similarity of users A and B, or in our case users 1 and 2, overlaps with the similarity calculated in the book (namely 0.38). Same goes for user A and C, in our case users 1 and 3 (namely 0.32).

We now apply the same distance measure to the movie dataset. 
## Movie Dataset Cosine Similarities
```{r}
cos_sim_movies <- train_m_wide / sqrt(rowSums(train_m_wide * train_m_wide))
cos_sim_movies <- cos_sim_movies %*% t(cos_sim_movies)
dim(cos_sim_movies)
```

# Normalizing Ratings

We follow the books example and subtract the average rating of each user from their ratings. Note that for user 4, the average rating was 3, so if we subtract it from all their ratings, we obtain 2 zeroes. 
## Example 9.9
```{r}
utility_matrix[utility_matrix == 0] <- NA #for the calculation, zero's are treated as missing values
utility_matrix
norm_utility_matrix <- t(apply(utility_matrix,1,norm<-function(x) return (x-mean(x,na.rm=T)))) #this way of setting up the normalization takes the matrices transpose, hence we transpose the matrix once more
norm_utility_matrix
```

We now set the NA's back to zero and compute cosine similarities. We see that the similarity of user 4 is always NaN, since the row of this user only contains zeroes. 
```{r}
norm_utility_matrix[is.na(norm_utility_matrix)] = 0 
cos_sim_norm_utility_m <- norm_utility_matrix / sqrt(rowSums(norm_utility_matrix * norm_utility_matrix))
cos_sim_norm_utility_m <- cos_sim_norm_utility_m %*% t(cos_sim_norm_utility_m)
cos_sim_norm_utility_m
```

## Movie Dataset

We now apply the same normalizing procedure as before and compute cosine similarities for the movie dataset. 
```{r}
#normalize over rows
train_m_wide[train_m_wide == 0] <- NA
norm_train_m_wide <- t(apply(train_m_wide,1,norm<-function(x) return (x-mean(x,na.rm=T))))
norm_train_m_wide[is.na(norm_train_m_wide)] = 0 
cos_sim_norm_movies <- norm_train_m_wide / sqrt(rowSums(norm_train_m_wide * norm_train_m_wide))
cos_sim_norm_movies <- cos_sim_norm_movies %*% t(cos_sim_norm_movies)
dim(cos_sim_norm_movies)
```

# Recommending movies
In the names dataframe, the id's for every movie-title are contained, search for the movie you want to have similar movies to. Let's see what the recommendation for Pretty Woman (1990) are. Its movie code is 597. Feel free to change the id, but keep in mind that any ID over 1000 is not in the dataset for the recommender. 

## Jaccard Recommendations

```{r}
#Remember we stored the jaccard distances in the variable jaccard_dist
movie_code <- 597 #597 is the movieId for Pretty Woman
new_movie_code <- max(df[df[,"movieId"]==movie_code,'new_movie_ID'])
jaccard_recs <- data.frame(jaccard_dist[,new_movie_code]) %>% #find column with of Pretty Woman, in other words the column's names is '231'
mutate(id = row_number()) %>%
  inner_join(names, by=c('id'='new_movie_ID')) %>%
  arrange_at('jaccard_dist...new_movie_code.') #Remember, in this case smaller distance means more similarity, so we sort ascendingly
jaccard_recs[1:10,]
```

## Cosine Recommnendations
```{r}
movie_code <- 597
new_movie_code <- max(df[df[,"movieId"]==movie_code,'new_movie_ID'])
cos_recs <- data.frame(cos_sim_movies[,new_movie_code]) %>%
  mutate(id = row_number()) %>%
  inner_join(names, by=c('id'='new_movie_ID')) %>%
  arrange_at('cos_sim_movies...new_movie_code.',desc) #In this case, higher cosine similarity means more similarity
cos_recs[1:10,]
```

## Normalized Cosine Recommendations
```{r}
movie_code <- 597
new_movie_code <- max(df[df[,"movieId"]==movie_code,'new_movie_ID'])
norm_cos_recs <- data.frame(cos_sim_norm_movies[,new_movie_code]) %>%
  mutate(id = row_number()) %>%
  inner_join(names, by=c('id'='new_movie_ID')) %>%
  arrange_at('cos_sim_norm_movies...new_movie_code.',desc) #In this case, higher cosine similarity means more similarity
norm_cos_recs[1:10,]
```

# Making predictions

Now we will create predictions for the users in the test set. For each movie that we left out, we select the _n_-most similar movies rated by the user. In our case we've set _n_ equal to 5. When we've found these _n_-most similar movies, we multiply the similarity score with the rating that the user gave to these movies. Then we scale the prediction with the sum of similarities. The formal calculation is as follows:
  $\textit{P}_{u,i} = \frac{\sum_{\textit{all similar items}, n}S_{i,n}\cdot\textit{R}_u,n}{\sum_{\textit{all similar items}, n}(|S_{i,n}|)}$^[http://www.cs.carleton.edu/cs_comps/0607/recommend/recommender/itembased.html]
```{r}
n<-5
pred_true <- matrix(ncol = 2, nrow = 0) #empty matrix which we will fill in the for-loop below
colnames(pred_true) <- c('true_rating','pred_rating')
for (row in 1:nrow(test_df)){
  #first things first, we create variables for the thing we want to predict
  user <- test_df[[row,1]] #user_id
  movieId_to_predict <- test_df[[row,2]] #movie_id_to_predict, the movieid which we will make a prediction for
  true_rating <- test_df[[row,3]] #true value, the prediction we make with movie_id_to_predict should be as close as possible
  new_movieId_to_predict <- test_df[[row,4]] #reindexed value for the movie_id
  new_user_ID <- test_df[[row,5]] #reindexed user_id
  sim_movies <- cos_sim_movies[new_movieId_to_predict,] #we find the row that corresponds to the movie_id_to_predict
  ratings_user <- t(train_m_wide)[new_user_ID,] #then we take the ratings for the user
  ratings_and_similarities <- as.data.frame(cbind(sim_movies, ratings_user)) #we bind the two vectors together
  ratings_and_similarities <- ratings_and_similarities[complete.cases(ratings_and_similarities),] #we delete all rows for movies that the user did not rate
  most_similar_n <- ratings_and_similarities[order(ratings_and_similarities$sim_movies, decreasing = T),][1:n,] #we sort the similarities from most similar to least similar
  #In this case, we set n equal to 5
  pred_rating <- most_similar_n %>%
    {.[,1]*.[,2]} %>% #Then we multiply the two columns
    sum/sum(most_similar_n$sim_movies) #Scale with the sum of similarities
  #now we have a predicted rating
  to_add <- as.matrix(cbind(true_rating,pred_rating)) #We add the prediction to the empty matrix
  pred_true<-rbind(pred_true, to_add)
}
#It results in the following RMSE
rmse(actual = pred_true[,1], predicted = pred_true[,2])
```