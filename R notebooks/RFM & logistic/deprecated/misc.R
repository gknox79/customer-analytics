
### Boosting

```{r}
library(gbm)

set.seed(19103)
ebeer_boost<-gbm(as.numeric(respmail)-1 ~ . -mailing, data=ebeer, distribution = "bernoulli", n.trees = 5000, interaction.depth = 4, shrinkage = .01)

summary(ebeer_boost)                 

par(mfrow=c(1,3))
a<-plot(ebeer_boost, i="M", type = "response")
b<-plot(ebeer_boost, i="F", type = "response")
c<-plot(ebeer_boost, i="student", type = "response")
print(a, position = c(0, 0, 0.35, 1), more = TRUE)
print(b, position = c(0.33, 0, .7, 1), more = TRUE)
print(c, position = c(0.67, 0, 1, 1))

```
