n = 5000
c = 1.5
m = 50
brk = c/m
x = 175


p_hat<-x/n
p_hat_se<-sqrt(p_hat*(1-p_hat)/n)

p_hat + c(-1,1)*2*p_hat_se


xx <- seq(.02,.05,length=1000)

par(mai=c(.9,.8,.2,.2))
plot(xx, dnorm(xx, p_hat, p_hat_se), type="l", col="royalblue", lwd=1.5,
     xlab="average response rate", ylab="density")
abline(v=brk)

pnorm(brk,p_hat,p_hat_se)


data = c(rep(1, 175), rep(0, 5000-175))

#nonparametric bootstrap
B <- 10000
mub <- c()
for (b in 1:B){
  samp_b = sample.int(length(data), replace=TRUE)
  mub <- c(mub, mean(data[samp_b]))
}
sd(mub)

par(mai=c(.8,.8,.2,.2))


hist(mub, main="", xlab="average response rate",
     col=8, border="grey90", freq=FALSE)
lines(xx, dnorm(xx, p_hat, p_hat_se), col="royalblue", lwd=1.5)
abline(v=brk)

sum(mub<brk)/B



#Bayesian prior

# Plot the prior
xx=seq(0,1,length=1000)

prior_a = 1
prior_b = 36

plot(xx, y=dbeta(xx, shape1=prior_a, shape2 = prior_b), 
     type="l", col="black", xlab="response rate", ylab="prior density")

post_a = prior_a + x
post_b = prior_b + n - x

plot(xx, y=dbeta(xx, shape1=post_a, shape2 = post_b), 
     type="l", col="black", xlab="response rate", ylab="posterior density")

xx=seq(0,.1,length=1000)

plot(xx, y=dbeta(xx, shape1=post_a, shape2 = post_b), 
     type="l", col="black", xlab="response rate", ylab="posterior density")
lines(xx, y=dbeta(xx, shape1=prior_a, shape2 = prior_b), type="l", col="gray")
abline(v=brk)
legend("topright", col=c("black", "gray"), legend=c("posterior", "prior"), bty="n", lty=1)
set.seed(19312)
post_draws<-rbeta(B,post_a,post_b)
prob = sum(post_draws<brk)/B
text(x = .02,y= 100, paste("P(p < .03) = ", round(prob,3) ))


# compare

xx=seq(0,1,length=1000)

prior_a = 1
prior_b = 1

n=100
x_A=.21*n
x_B=.38*n

plot(xx, y=dbeta(xx, shape1=prior_a, shape2 = prior_b), 
     type="l", col="black", xlab="response rate", ylab="prior density")

post_A_a = prior_a + x_A
post_A_b = prior_b + n - x_A

post_B_a = prior_a + x_B
post_B_b = prior_b + n - x_B

xx=seq(0,1,length=1000)

plot(x=xx, y=dbeta(xx, shape1=post_A_a, shape2 = post_A_b), 
     type="l", col="blue", xlab="conversion rate", ylab="posterior density")
lines(x=xx, y=dbeta(xx, shape1=post_B_a, shape2 = post_B_b), col="red")
lines(x=xx, y=dbeta(xx, shape1=prior_a, shape2 = prior_b), col="gray")
legend("topright", col=c("blue", "red", "gray"), legend=c("posterior for A", "posterior for B", "prior"), bty="n", lty=1)




set.seed(19312)
p_A <-rbeta(n = 10000, shape1 = 1+21, shape2 = 1+100???21)
p_B <-rbeta(n = 10000, shape1 = 1+38, shape2 = 1+100-38)

sum(p_A<p_B)/10000

set.seed(19312)
B<-10000
post_draws_A<-rbeta(B,post_A_a,post_A_b)
post_draws_B<-rbeta(B,post_B_a,post_B_b)
prob = sum(post_draws_B>post_draws_A)/B
text(x = .6,y= 6, paste("P(p_A < p_B) = ", round(prob,3) ))

round(prob,3)


pnorm(0,mean = 0,sd=1)

pnorm(8,mean=5, sd=1)

pnorm(2, mean=5, sd=1)


mean = 175/5000

((mean)*(1-mean)/5000)^(1/2)

sqrt((mean*(1-mean))/5000)

pnorm(.03, mean = mean, sd = std_err)

round(0.48278, digits = 2)


round(29347.84021, digits = 0)

# Normal-Normal



set.seed(19312)
group <- c(rep("A", 500), rep("B", 500)) 
time_on_site <- c(rnorm(500, mean=5.2, sd=2), rnorm(500, mean=5.4, sd=2.2))
test_data <- data.frame(group, time_on_site)
rm(group, time_on_site)


test_data %>% 
  group_by(group) %>% summarize(mean=mean(time_on_site), sd=sd(time_on_site), n=n())


n_A <- sum(test_data$group=="A")
n_B <- sum(test_data$group=="B")
s <- sd(test_data$time_on_site)
post_mean_A <- mean(test_data[test_data$group=="A", "time_on_site"])
post_mean_B <- mean(test_data[test_data$group=="B", "time_on_site"])
post_sd_A <- (1/100^2 + n_A/s^2)^-(1/2)
post_sd_B <- (1/100^2 + n_B/s^2)^-(1/2)

post_mean_A = post_sd_A^2 * (mean(test_data[test_data$group=="A", "time_on_site"])*n_A/s^2)

xx=seq(10,60,length=1000)

plot(x=(500:600)/100, y=dnorm((500:600)/100, mean=post_mean_A, sd=post_sd_A), 
     type="l", col="blue", xlab="mean time-on-site (m)", ylab="posterior density")
lines(x=(500:600)/100, y=dnorm((500:600)/100, mean=post_mean_B, sd=post_sd_B), col="red")
lines(x=(500:600)/100, y=dnorm((500:600)/100, mean=0, sd=100), col="gray")
legend("topright", col=c("blue", "red", "gray"), legend=c("posterior for A", "posterior for B", "prior"), bty="n", lty=1)
post_mean_diff <- post_mean_B - post_mean_A
post_sd_diff <- sqrt(post_sd_B^2 + post_sd_A^2)
#Once we have the distribution for the difference in the mean time-on-site, we can compute the probability that the mean of B is greater than the mean of A. 
prob=1-pnorm(0, mean=post_mean_diff, sd=post_sd_diff)
text(x = 5.75,y= 2, paste("P(m_A < m_B) = ", round(prob,3) ))

plot(x=(-50:60)/100, y=dnorm((-50:60)/100, mean=post_mean_diff, sd=post_sd_diff), 
     type="l", col="black", 
     xlab="difference in mean time-on-site (m)", ylab="posterior density")
abline(v=0)
text(-0.25, 2.9, "A has higher mean time-on-site")
text(0.35, 2.9, "B has higher mean time-on-site")
