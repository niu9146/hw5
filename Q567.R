#TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST





# The following code analyzes the federalist papers
#############################

#################
# Setup
#################

# make sure R is in the proper working directory
# note that this will be a different path for every machine
# setwd("~/Documents/academic/teaching/STAT_W4240_2014_SPRG/dropbox/Homework/hw04")

# first include the relevant libraries
# note that a loading error might mean that you have to
# install the package into your R distribution.
# Use the package installer and be sure to install all dependencies
library(tm)
library(SnowballC)
library(rpart)
library(glmnet)

#################
# Problem 5a
#################



setwd("~/Documents/ColumbiaUniversity/First Semester/Data Mining/HW5")

library(tm)
library(SnowballC)
library(rpart)
library(glmnet)

source('hw4.R')

hamilton.train=read.directory('fp_hamilton_train_clean')
hamilton.test=read.directory('fp_hamilton_test_clean')
madison.train=read.directory('fp_madison_train_clean')
madison.test=read.directory('fp_madison_test_clean')


combined1=c(hamilton.train,hamilton.test)
combined2=c(madison.train,madison.test)
combined_total=c(combined1,combined2)
dictionary=make.sorted.dictionary.df(combined_total)

dtm.hamilton.train=make.document.term.matrix(hamilton.train,dictionary)
dtm.hamilton.test=make.document.term.matrix(hamilton.train,dictionary)
dtm.madison.train=make.document.term.matrix(madison.train,dictionary)
dtm.madison.test=make.document.term.matrix(madison.test,dictionary)
##########################################

##########################################
# SET UP
#add labels; 1 for hamilton, 0 for madison

hamilton.train.labels = cbind(dtm.hamilton.train, as.vector(rep(1, dim(dtm.hamilton.train)[1])))
madison.train.labels = cbind(dtm.madison.train, as.vector(rep(0, dim(dtm.madison.train)[1])))
hamilton.test.labels = cbind(dtm.hamilton.test, as.vector(rep(1, dim(dtm.hamilton.test)[1])))
madison.test.labels = cbind(dtm.madison.test, as.vector(rep(0, dim(dtm.madison.test)[1])))

# data frame of all training data
training.data = data.frame(rbind(hamilton.train.labels, madison.train.labels))
# data frame of all testing data
testing.data = data.frame(rbind(hamilton.test.labels, madison.test.labels))

# make the column names the actual words
colnames(training.data)<-c(as.vector(dictionary$word), 'y')
colnames(testing.data)<-c(as.vector(dictionary$word), 'y')

attach(training.data)

# Tree classification using Gini coefficient
tree.federalist = rpart(y ~ ., data=training.data, method="class")
federalist.fit.gini = predict(tree.federalist, testing.data, type="class")
print(federalist.fit.gini)

# Errors
gini.proportion.correct = sum(testing.data$y == federalist.fit.gini)/length(testing.data$y)
gini.false.negatives = sum(federalist.fit.gini[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
gini.false.positives = sum(federalist.fit.gini[(dim(dtm.hamilton.test)[1]+1):length(federalist.fit.gini)] == 1)/dim(dtm.madison.test)[1]

gini.proportion.correct
gini.false.negatives
gini.false.positives

# plot gini tree
par(xpd = TRUE)
plot(tree.federalist)
text(tree.federalist, use.n=TRUE)
filename = 'hw05_05a.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

##########################################

#################
# Problem 5b
#################

##########################################
# Tree classification using information gain
tree.federalist.ig = rpart(y ~ ., data=training.data, method="class", parms=list(split='information'))
#tree.federalist.ig = rpart(training.data, data=training.data, method="class", parms=list(split='information'))

federalist.fit.ig = predict(tree.federalist.ig, testing.data, type="class")

print(federalist.fit.ig)

# Errors
ig.proportion.correct = sum(testing.data$y == federalist.fit.ig)/length(testing.data$y)
ig.false.negatives = sum(federalist.fit.ig[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
ig.false.positives = sum(federalist.fit.ig[(dim(dtm.hamilton.test)[1]+1):length(federalist.fit.ig)] == 1)/dim(dtm.madison.test)[1]

ig.proportion.correct
ig.false.negatives
ig.false.positives

# plot information gain tree
par(xpd = TRUE)
plot(tree.federalist.ig)
text(tree.federalist.ig, use.n=TRUE)
filename = 'hw05_05b.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()


##########################################

#################
# Problem 6b
#################

##########################################

# scale training data
without_last_one = dim(training.data)[2] - 1
training.data.scaled = scale(rbind(hamilton.train.labels, madison.train.labels)[,1:without_last_one])
# convert all NA to 0
training.data.scaled[is.na(training.data.scaled)] = 0

# scale testing data
without_last_one = dim(testing.data)[2] - 1
testing.data.scaled = scale(testing.data[,1:without_last_one])
# convert all NA to 0
testing.data.scaled[is.na(testing.data.scaled)] = 0

# ridge regression model
federalist.fit.ridge.lambda = cv.glmnet(training.data.scaled, y, family="binomial", alpha=0)
federalist.fit.ridge = glmnet(training.data.scaled, y, family="binomial", alpha=0, lambda=federalist.fit.ridge.lambda$lambda.min)

# predict test set using ridge regression model
federalist.pred.ridge = predict(federalist.fit.ridge, testing.data.scaled, type="class")
print(federalist.pred.ridge)

# Errors
ridge.proportion.correct = sum(testing.data$y == federalist.pred.ridge)/length(testing.data$y)
ridge.false.negatives = sum(federalist.pred.ridge[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
ridge.false.positives = sum(federalist.pred.ridge[(dim(dtm.hamilton.test)[1]+1):length(federalist.pred.ridge)] == 1)/dim(dtm.madison.test)[1]

ridge.proportion.correct
ridge.false.negatives
ridge.false.positives

#coefficients
ridge.beta = federalist.fit.ridge$beta

# top 10 words
# as.vector(dictionary$word)
ridge.words = list()
ridge.word_indices = matrix(nrow=10, ncol=2)
for (i in seq(1:10)){
    word_index = which(abs(ridge.beta) == max(abs(ridge.beta)))
    ridge.words = append(ridge.words, as.vector(dictionary$word)[word_index])
    ridge.word_indices[i,] = c(word_index, ridge.beta[word_index])
    # set to zero so we don't select the same word twice
    ridge.beta[word_index] = 0
}
# words
ridge.words
# indices and coefficients
ridge.word_indices

##########################################

#################
# Problem 6c
#################

##########################################

# lasso regression model
federalist.fit.lasso.lambda = cv.glmnet(training.data.scaled, y, family="binomial", alpha=1)
federalist.fit.lasso = glmnet(training.data.scaled, y, family="binomial", alpha=1, lambda=federalist.fit.lasso.lambda$lambda.min)

# predict test set using lasso regression model
federalist.pred.lasso = predict(federalist.fit.lasso, testing.data.scaled, type="class")
print(federalist.pred.lasso)

# Errors
lasso.proportion.correct = sum(testing.data$y == federalist.pred.lasso)/length(testing.data$y)
lasso.false.negatives = sum(federalist.pred.lasso[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
lasso.false.positives = sum(federalist.pred.lasso[(dim(dtm.hamilton.test)[1]+1):length(federalist.pred.lasso)] == 1)/dim(dtm.madison.test)[1]

lasso.proportion.correct
lasso.false.negatives
lasso.false.positives

#coefficients
lasso.beta = federalist.fit.lasso$beta

# top 10 words
# as.vector(dictionary$word)
lasso.words = list()
lasso.word_indices = matrix(nrow=10, ncol=2)
for (i in seq(1:10)){
    word_index = which(abs(lasso.beta) == max(abs(lasso.beta)))
    lasso.words = append(lasso.words, as.vector(dictionary$word)[word_index])
    lasso.word_indices[i,] = c(word_index, lasso.beta[word_index])
    # set to zero so we don't select the same word twice
    lasso.beta[word_index] = 0
}
# words
lasso.words
# indices and coefficients
lasso.word_indices

##########################################

#################
# Problem 7b
#################

##########################################
# Mutual information

#calculate probabiliites using make.log.pvec
make.log.pvec <- function(dtm,mu){
    # Sum up the number of instances per word
    pvec.no.mu <- colSums(dtm)
    # Sum up number of words
    n.words <- sum(pvec.no.mu)
    # Get dictionary size
    dic.len <- length(pvec.no.mu)
    # Incorporate mu and normalize
    log.pvec <- log(pvec.no.mu + mu) - log(mu*dic.len + n.words)
    return(log.pvec)
}

D = dim(dictionary)[1]
mu = 1/D

hamilton.log.probs = make.log.pvec(dtm.hamilton.train, mu)
madison.log.probs = make.log.pvec(dtm.madison.train, mu)

hamilton.count = dim(dtm.hamilton.train)[1]
madison.count = dim(dtm.madison.train)[1]
total.count = dim(training.data)[1]

hamilton.prob = hamilton.count/total.count
madison.prob = 1 - hamilton.prob

mutual_information = vector()
for (i in 1:D) {
    prob.x = (exp(hamilton.log.probs[i]) * hamilton.count + exp(madison.log.probs[i]) * madison.count)/total.count
    hamilton_info = exp(hamilton.log.probs[i]) * hamilton.prob *
        log(exp(hamilton.log.probs[i])/prob.x) +
        (1-exp(hamilton.log.probs[i])) * hamilton.prob *
        log((1-exp(hamilton.log.probs[i]))/(1-prob.x))
    madison_info = exp(madison.log.probs[i]) * madison.prob * 
        log(exp(madison.log.probs[i])/prob.x) +
        (1-exp(madison.log.probs[i])) * madison.prob * 
        log((1-exp(madison.log.probs[i]))/(1-prob.x))
    mutual_information[i] = hamilton_info + madison_info
}


# for the selected n's, choose the top n features
# then recompute the errors for the classifiers in 5 and 6
n_levels = c(200, 500, 1000, 2500)

# errors matrices
# rows are gini, information gain, ridge, lasso
# columns are n
proportion.correct = matrix(nrow=4, ncol=4)
false.negatives = matrix(nrow=4, ncol=4)
false.positives = matrix(nrow=4, ncol=4)

prev_n = 0
mutual_information_copy = mutual_information
top_words_indices = vector()
for (n_index in 1:length(n_levels)) {
    n = n_levels[n_index]
    print(n)

    for (i in (prev_n+1):n) {
        top_word_index = which(mutual_information_copy == max(mutual_information_copy))
        # if more than one word is max, just choose the first
        # the others will come up in the next iteration
        if (length(top_word_index) > 1) {
            top_word_index = top_word_index[1]
        }
        top_words_indices[i] = top_word_index
        # set to zero so we don't select the same word twice
        mutual_information_copy[top_word_index] = 0
    }

    # Set up using limited dictionary
    new_dictionary = as.vector(dictionary$word)[top_words_indices]
    
    hamilton.train.labels = cbind(dtm.hamilton.train[,top_words_indices], as.vector(rep(1, dim(dtm.hamilton.train)[1])))
    madison.train.labels = cbind(dtm.madison.train[,top_words_indices], as.vector(rep(0, dim(dtm.madison.train)[1])))
    hamilton.test.labels = cbind(dtm.hamilton.test[,top_words_indices], as.vector(rep(1, dim(dtm.hamilton.test)[1])))
    madison.test.labels = cbind(dtm.madison.test[,top_words_indices], as.vector(rep(0, dim(dtm.madison.test)[1])))

    # data frame of all training data
    training.data = data.frame(rbind(hamilton.train.labels, madison.train.labels))
    # data frame of all testing data
    testing.data = data.frame(rbind(hamilton.test.labels, madison.test.labels))

    # make the column names the actual words
    colnames(training.data)<-c(as.vector(new_dictionary), 'y')
    colnames(testing.data)<-c(as.vector(new_dictionary), 'y')

    # Tree classification using Gini coefficient
    tree.federalist = rpart(training.data$y ~ ., data=training.data, method="class")
    federalist.fit.gini = predict(tree.federalist, testing.data, type="class")

    # Errors for Gini coefficient
    proportion.correct[1,n_index] = sum(testing.data$y == federalist.fit.gini)/length(testing.data$y)
    false.negatives[1,n_index] = sum(federalist.fit.gini[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
    false.positives[1,n_index] = sum(federalist.fit.gini[(dim(dtm.hamilton.test)[1]+1):length(federalist.fit.gini)] == 1)/dim(dtm.madison.test)[1]

    # Tree classification using information gain
    tree.federalist.ig = rpart(training.data$y ~ ., data=training.data, method="class", parms=list(split='information'))
    

    federalist.fit.ig = predict(tree.federalist.ig, testing.data, type="class")

    # Errors for information gain
    proportion.correct[2,n_index] = sum(testing.data$y == federalist.fit.ig)/length(testing.data$y)
    false.negatives[2,n_index] = sum(federalist.fit.ig[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
    false.positives[2,n_index] = sum(federalist.fit.ig[(dim(dtm.hamilton.test)[1]+1):length(federalist.fit.ig)] == 1)/dim(dtm.madison.test)[1]

    # Ridge and Lasso Regression Models
    # scale training data
    all_but_one = dim(training.data)[2] - 1
    training.data.scaled = scale(rbind(hamilton.train.labels, madison.train.labels)[,1:all_but_one])
    # convert all NA to 0
    training.data.scaled[is.na(training.data.scaled)] = 0

    # scale testing data
    all_but_one = dim(testing.data)[2] - 1
    testing.data.scaled = scale(testing.data[,1:all_but_one])
    # convert all NA to 0
    testing.data.scaled[is.na(testing.data.scaled)] = 0

    # ridge regression model
    federalist.fit.ridge.lambda = cv.glmnet(training.data.scaled, training.data$y, family="binomial", alpha=0)
    federalist.fit.ridge = glmnet(training.data.scaled, training.data$y, family="binomial", alpha=0, lambda=federalist.fit.ridge.lambda$lambda.min)

    # predict test set using ridge regression model
    federalist.pred.ridge = predict(federalist.fit.ridge, testing.data.scaled, type="class")

    # Errors for ridge regression
    proportion.correct[3,n_index] = sum(testing.data$y == federalist.pred.ridge)/length(testing.data$y)
    false.negatives[3,n_index] = sum(federalist.pred.ridge[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
    false.positives[3,n_index] = sum(federalist.pred.ridge[(dim(dtm.hamilton.test)[1]+1):length(federalist.pred.ridge)] == 1)/dim(dtm.madison.test)[1]

    # lasso regression model
    federalist.fit.lasso.lambda = cv.glmnet(training.data.scaled, training.data$y, family="binomial", alpha=1)
    federalist.fit.lasso = glmnet(training.data.scaled, training.data$y, family="binomial", alpha=1, lambda=federalist.fit.lasso.lambda$lambda.min)

    # predict test set using lasso regression model
    federalist.pred.lasso = predict(federalist.fit.lasso, testing.data.scaled, type="class")

    # Errors
    proportion.correct[4,n_index] = sum(testing.data$y == federalist.pred.lasso)/length(testing.data$y)
    false.negatives[4,n_index] = sum(federalist.pred.lasso[1:dim(dtm.hamilton.test)[1]] == 0)/dim(dtm.hamilton.test)[1]
    false.positives[4,n_index] = sum(federalist.pred.lasso[(dim(dtm.hamilton.test)[1]+1):length(federalist.pred.lasso)] == 1)/dim(dtm.madison.test)[1]

    prev_n = n
}

proportion.correct
false.negatives
false.positives


#Plot and save proportion correct
plot(n_levels, proportion.correct[1,], main="Proportion Correct",
    xlab="n", ylab="Proportion Correct")
lines(n_levels, proportion.correct[2,], col=2)
lines(n_levels, proportion.correct[3,], col=3)
lines(n_levels, proportion.correct[4,], col=4)
filename = 'hw05_pc.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

#Plot and save false negatives
plot(n_levels, false.negatives[1,], main="False Negatives",
    xlab="n", ylab="False Negatives")
lines(n_levels, false.negatives[2,], col=2)
lines(n_levels, false.negatives[3,], col=3)
lines(n_levels, false.negatives[4,], col=4)
filename = 'hw05_fn.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

#Plot and save false postives
yrange = range(c(false.positives[2:4,]))
plot(n_levels, false.positives[1,], main="False Positives",
    xlab="n", ylab="False Positives", ylim=yrange)
lines(n_levels, false.positives[2,], col=2)
lines(n_levels, false.positives[3,], col=3)
lines(n_levels, false.positives[4,], col=4)
filename = 'hw05_fp.png'
dev.copy(device=png, file=filename, height=600, width=800)
dev.off()

##########################################

#################
# End of Script
#################