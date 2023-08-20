### Packages required ###
#For python interface
require("reticulate")

#For general tidyness
library("dplyr")
require("ggplot2")
require("ggpubr")
require("tidyverse")
require("purrr")
require("maditr") # dcast

#For mixed two way anova
require("rstatix")

#For Machine learning data spliting
require("caTools")
library("caret")

# For machine learning
require("randomForest") # Random Forest
require("e1071") # Support vector machine
library("pls") # PLSR 

# For ploting confusion matrix #
library("cvms")

# FOR RMSEP 
library("Metrics")
# Not needed ?
#require("clusterer")
#reuire("vegan")
#require("ChemoSpec")

### Load data from python pickle ###
source_python("~/Documents/MastersProject/code/pickle_reader.py")
pickledata <- read_pickle_file("~/Documents/MastersProject/code/powerspec_full.pk1")

pickledata <- pickledata[,5:1515]

### Data cleaning ###
### Encode binary choices ###
# EXP = 1 
# Yes = 1
pickledata$C_X <- ifelse(pickledata$C_X == "Experimental", 1, 0)
pickledata$pol_stat <- ifelse(pickledata$pol_stat == "Y", 1, 0)
pickledata$pol_outcome <- ifelse(pickledata$pol_outcome == "Y", 1, 0)

### Convert Log fourier columns ###
pickledata$flower_age <- as.numeric(pickledata$flower_age)
pickledata$flower_start <- as.numeric(pickledata$flower_start)
cols_4_log <- colnames(pickledata)[12:1511]
pickledata[cols_4_log] <- log10(pickledata[cols_4_log])

### Add new column for two flower varieties ###
pickledata$variety <- ifelse(pickledata$plant_id > 49, 1, 0)
### Add anomaly column ###
pickledata$anomaly <- ifelse(pickledata$C_X != pickledata$pol_outcome, 1, 0)

### Let clean the data ###
#remove anomalies and variety 0

DF_clean <- filter(pickledata, anomaly == 0 & variety == 1)


### Filter to get only flowers with all 6 time data points ###
DF_xtra_clean <- filter(DF_clean, flower_age %in% c(0,2,4,6,8,24,26,28,30,32))
DF_xtra_clean <- as.data.frame(cbind(DF_xtra_clean[,1],DF_xtra_clean[,4:5],DF_xtra_clean[,10],DF_xtra_clean[,12:length(DF_xtra_clean)]))

colnames(DF_xtra_clean)[1] <- "C_X"
colnames(DF_xtra_clean)[4]<-"flower_age"

DF_xtra_clean$unique_id <- factor(DF_xtra_clean$unique_id)
DF_xtra_clean$C_X <- factor(DF_xtra_clean$C_X)
DF_xtra_clean$pol_stat <- factor(DF_xtra_clean$pol_stat)
DF_xtra_clean$flower_age <- factor(DF_xtra_clean$flower_age)

data_wide <- dcast(DF_xtra_clean, C_X + unique_id ~ flower_age, value.var = "pol_stat")
data_wide <- data_wide %>% drop_na()

# THIS IS NOW MY SQUEKY CLEAN DATA #
DF2 <- pickledata %>% filter(unique_id %in% data_wide$unique_id & flower_age %in% c(0,2,4,6,8,24,26,28,30,32))
DF2$unique_id <- factor(DF2$unique_id)
DF2 <- DF2[,c(1,4:5,10,12:1511)]
DF2$C_X <- factor(DF2$C_X)
DF2$pol_stat <- factor(DF2$pol_stat)

### Filter pickle data to what we've used and save for later ###
save(DF2, file = "~/Documents/MastersProject/code/clean_data.Rda")

# Just power spec for PCA analysis
DF_SPEC <- DF2[,5:1504]
DF_SPEC <- scale(DF_SPEC)
#cov_matrix <- cov(DF_SPEC)
# Check for normality 
# ggqqplot(DF_PCA[,1])

### PEFORM PCA ANALYSIS ###
pca <- prcomp(DF_SPEC)

### Looking at the variance explained by each PC ###
eigs <- pca$sdev^2
eigs_sum <- sum(eigs)
for (iter in 1:length(eigs)){
  eigs[iter] <- eigs[iter] / eigs_sum
}
eig_cumulative<-round(cumsum(eigs)*100, digits= 1)

### Create a df for plotting ###
PC_loadings <- paste(pca$x[,1:15])
PC_X <- rep(1:15, each=1580)
plot_df2 <- as.data.frame(cbind(PC_X,PC_loadings))
plot_df2$PC_X <- as.integer(plot_df2$PC_X)
plot_df2$PC_loadings <- as.double(plot_df2$PC_loadings)



PCA_plot2 <- ggplot(data=plot_df2, aes(x=PC_X, y=PC_loadings))+ 
  geom_point(colour="blue", alpha=0.3, size=4, shape=1)+
  geom_text(data = as.data.frame(eig_cumulative[1:15]), aes(x=1:15, y = 0, label=eig_cumulative[1:15]), nudge_x = 0.5, vjust = -0.8)+
  annotate("label", x = 12, y= 225, label="Cumulative percent varince shown next to Principle Component")+
  scale_x_continuous("Component", breaks = 1:15)+
  labs(x="Component", y = "Score")+
  geom_hline(yintercept = 0, linetype="dashed")+
  theme_linedraw()+
  theme(panel.grid.minor = element_blank())

# Plot to PDF
pdf("~/Documents/MastersProject/images/plots/PCA_axis_variance.pdf", width = 10, height = 6.5)  
print(PCA_plot2)
dev.off()

  
### Take loadings into dataframe ###
ANOVA_DF <- as.data.frame(cbind(DF2[1:4],pca$x[,1:15]))
colnames(ANOVA_DF)[1] <- "treatment"

### Run two way mixed anova ###
t1 <- anova_test(data = ANOVA_DF, dv = PC1, wid = unique_id, between = treatment, within = flower_age)

get_anova_table(t1, correction = "GG") # Correction for sphericity - check this

t2 <- anova_test(data = ANOVA_DF, dv = PC2, wid = unique_id, between = treatment, within = flower_age)
get_anova_table(t2, correction = "GG") # Correction for sphericity - check this

t3 <- anova_test(data = ANOVA_DF, dv = PC3, wid = unique_id, between = treatment, within = flower_age)
t4 <- anova_test(data = ANOVA_DF, dv = PC4, wid = unique_id, between = treatment, within = flower_age)
t5 <- anova_test(data = ANOVA_DF, dv = PC5, wid = unique_id, between = treatment, within = flower_age)
t6 <- anova_test(data = ANOVA_DF, dv = PC6, wid = unique_id, between = treatment, within = flower_age)
t7 <- anova_test(data = ANOVA_DF, dv = PC7, wid = unique_id, between = treatment, within = flower_age)
t8 <- anova_test(data = ANOVA_DF, dv = PC8, wid = unique_id, between = treatment, within = flower_age)
t9 <- anova_test(data = ANOVA_DF, dv = PC9, wid = unique_id, between = treatment, within = flower_age)
t10 <- anova_test(data = ANOVA_DF, dv = PC10, wid = unique_id, between = treatment, within = flower_age)
t11 <- anova_test(data = ANOVA_DF, dv = PC11, wid = unique_id, between = treatment, within = flower_age)
t12 <- anova_test(data = ANOVA_DF, dv = PC12, wid = unique_id, between = treatment, within = flower_age)
t13 <- anova_test(data = ANOVA_DF, dv = PC13, wid = unique_id, between = treatment, within = flower_age)
t14 <- anova_test(data = ANOVA_DF, dv = PC14, wid = unique_id, between = treatment, within = flower_age)
t15 <- anova_test(data = ANOVA_DF, dv = PC15, wid = unique_id, between = treatment, within = flower_age)



write.table(get_anova_table(t1, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=TRUE, sep=",")
write.table(get_anova_table(t2, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t3, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t4, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t5, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t6, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t7, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t8, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t9, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t10, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t11, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t12, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t13, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t14, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(get_anova_table(t15, correction = "GG"), "~/Documents/MastersProject/writeup/pca_anova.csv", col.names=FALSE, sep=",", append=TRUE)

### POST HOC test on PCS with flower_age ###
### PCs 1,3,4,5,6,7,9,14 ###
pht1 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC1 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht2 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC2 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht3 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC3 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht4 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC4 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht5 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC5 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht6 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC6 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht7 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC7 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht9 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC9 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht13 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC13 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
pht14 <- ANOVA_DF %>% group_by(flower_age) %>% anova_test(PC14 ~ treatment) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")


write.table(pht1, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=TRUE, sep=",")
write.table(pht2, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht3, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht4, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht5, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht6, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht7, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht9, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht13, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(pht14, "~/Documents/MastersProject/writeup/phoc_anova.csv", col.names=FALSE, sep=",", append=TRUE)

phtB1 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC1 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB2 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC2 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB3 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC3 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB4 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC4 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB5 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC5 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB6 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC6 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB7 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC7 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB9 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC9 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB13 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC13 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")
phtB14 <- ANOVA_DF %>% group_by(treatment) %>% anova_test(PC14 ~ flower_age) %>% get_anova_table() %>% adjust_pvalue(method = "bonferroni")

write.table(phtB1, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=TRUE, sep=",")
write.table(phtB2, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB3, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB4, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB5, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB6, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB7, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB9, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB13, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)
write.table(phtB14, "~/Documents/MastersProject/writeup/phoc_anova2.csv", col.names=FALSE, sep=",", append=TRUE)




#### MACHINE LEARNING ####

### RANDOM FOREST ### 
# Flower age continous so regression and plot #

### Set correct factors ###


set.seed(1234)
split <- sample.split(DF2$C_X, SplitRatio = 0.6)


train <- subset(DF2[1:1504], split == "TRUE")
test <- subset(DF2[1:1504], split == "FALSE")

classifier_RF <- randomForest(x = train[,5:1504], y = train$flower_age, ntree = 500 )

classifier_RF

y_pred_RF = predict(classifier_RF, newdata = test[,5:1504])

plot_age_RF <- as.data.frame(cbind(test[,4], y_pred_RF))

### Regression and plot ###
# Make pretty and plot trend line #
RF_reg <- lm(test[,4] ~ y_pred_RF)
plot(test[,4], y_pred_RF)
abline(RF_reg)

RMSE_age_rf <- rmse(y_pred_RF,test[,4])

RF_age_plot <- ggplot(data=plot_age_RF, aes(x=test[,4], y=y_pred_RF))+ 
  geom_boxplot(aes(group = test[,4]))+
  labs(x="Measured time since flower opening", y = "Predicted time since flower opening")+
  geom_abline(intercept = RF_reg$coefficients[[1]], slope = RF_reg$coefficients[[2]]) +
  annotate("label", x = 30, y= 3, label="R-squared: 0.4126 \n y = 1.141x -1.113")+
  theme_linedraw()+
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

pdf("~/Documents/MastersProject/images/plots/RF_age_regression.pdf", width = 10, height = 6.5)  
print(RF_age_plot)
dev.off()

### RF for days ###
set.seed(1234)
DF2$day <- factor(ifelse(DF2$flower_age > 23, 2, 1))

train.day <- subset(DF2[1:1505], split == "TRUE")
test.day <- subset(DF2[1:1505], split == "FALSE")

classifier_RF_day <- randomForest(x = train.day[,5:1504], y = train.day$day, ntree = 500 )

classifier_RF_day

y_pred_RF_day = predict(classifier_RF_day, newdata = test.day[,5:1504])

day_RF_mtx = table(y_pred_RF_day , test.day[,1505] )



### RF for treatment ##
treatment_RF <- randomForest(x = train[,5:1504], y = train$C_X, ntree = 500 )
y_pred_treat_RF = predict(treatment_RF, newdata = test[,5:1504])
treatment_RF_mtx = table(test[,1], y_pred_treat_RF)

treatment_RF_mtx

### Support vector machine ###

classifier_SVM <- svm(formula = C_X ~ ., data = train[,c(1,5:1504)], type = "C-classification", kernel = "radial")
y_pred_treat_SVM = predict(classifier_SVM, newdata = test[,5:1504])
treatment_SVM_mtx = table(test[,1], y_pred_treat_SVM)
treatment_SVM_mtx
#plot_confussion_mtx <- as.data.frame(confusion_mtx)

#colnames(plot_confussion_mtx)[2] <- "prediction"
#colnames(plot_confussion_mtx)[1] <- "target"
#conf_plot <- plot_confusion_matrix(plot_confussion_mtx, target_col ="target", prediction_col = "prediction", counts_col = "Freq")

### SVM experiment ###
DF3 <- filter(DF2, DF2$flower_age > 23)
split3 <- sample.split(DF3$C_X, SplitRatio = 0.6)
split

train3 <- subset(DF3[1:1504], split3 == "TRUE")
test3 <- subset(DF3[1:1504], split3 == "FALSE")

classifier_SVM3 <- svm(formula = C_X ~ ., data = train3[,c(1,5:1504)], type = "C-classification", kernel = "radial")
y_pred_treat_SVM3 = predict(classifier_SVM3, newdata = test3[,5:1504])
treatment_SVM_mtx3 = table(test3[,1], y_pred_treat_SVM3)
treatment_SVM_mtx3

### RF expreiment ###
classifier_RF3 <- randomForest( x = train3[,5:1504], y = train3$C_X, ntree = 500)
y_pred_treat_RF3 = predict(classifier_RF3, newdata = test3[,5:1504])
treatment_RF_mtx3 = table(test3[,1], y_pred_treat_RF3)







###READ CNN RESULTS IN FOR PLOTTING ####
source_python("~/Documents/MastersProject/code/pickle_reader.py")
cnn_results <- read_pickle_file("~/Documents/MastersProject/code/cnn_age_results.pk1")

CNN_reg <- lm(cnn_results$Actual_flower_age ~ cnn_results$Predicted_flower_age)

CNN_age_plot <- ggplot(data=cnn_results, aes(x=Actual_flower_age, y=Predicted_flower_age))+ 
  geom_boxplot(aes(group = Actual_flower_age))+
  labs(x="Measured time since flower opening", y = "Predicted time since flower opening")+
  geom_abline(intercept = CNN_reg$coefficients[[1]], slope = CNN_reg$coefficients[[2]]) +
  annotate("label", x = 2.5, y= 37.5, label="R-squared: 0.4169 \n y = 0.839x +4.068")+
  theme_linedraw()+
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

pdf("~/Documents/MastersProject/images/plots/CNN_age_regression.pdf", width = 10, height = 6.5)  
print(CNN_age_plot)
dev.off()
