library(rfm)
library(lubridate)

setwd("C:/Users/gknox/Dropbox/Customer Analytics/R notebooks/RFM & logistic")

cdNOW <- read.csv('./data/CDnow.txt', header=FALSE, sep = "")

colnames(cdNOW) <- c("CustomerID","InvoiceDate","Quantity","Amount")

cdNOW$InvoiceDate <-ymd(cdNOW$InvoiceDate)
cdNOW<-cdNOW[,c(-3)]
head(cdNOW)
summary(cdNOW)
analysis_date <- lubridate::as_date("1998-07-01")

rfm<-rfm_table_order(cdNOW, customer_id = CustomerID, order_date = InvoiceDate, revenue = Amount, analysis_date)

rfm_histograms(rfm)

respRFM<-aggregate(rfm$rfm$customer_id, by = list(rfm$rfm$rfm_score), FUN = "length")
