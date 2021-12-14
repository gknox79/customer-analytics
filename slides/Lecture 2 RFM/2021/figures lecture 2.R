c = 1.5
m = 50
brk = c/m


p_hat<-1/36
p_hat_se<-sqrt(p_hat*(1-p_hat)/36)

p_hat + c(-1,1)*2*p_hat_se


xx <- seq(0,.1,length=1000)

par(mai=c(.9,.8,.2,.2))
plot(xx, dnorm(xx, p_hat, p_hat_se), type="l", col="royalblue", lwd=1.5,
     xlab="response rate for segment 353", ylab="density")
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
hist(mub, main="", xlab="response rate for segment 353",
     col=8, border="grey90", freq=FALSE)
lines(xx, dnorm(xx, p_hat, p_hat_se), col="royalblue", lwd=1.5)
abline(v=brk)

sum(mub<brk)/B
