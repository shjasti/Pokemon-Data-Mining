train <- sample(1:dim(dat)[1], dim(dat)[1] /2)
test <- -train

dat.train <- dat[train, ]
dat.test <- dat[test, ]


m1 <- gam(log(weight.num) ~ s(hp, 3) + s(attack, 3) + s(defense, 3) + s(spattack, 3) + s(spdefense, 3) + s(speed, 3), data = dat, subset = train)
mean((log(dat$weight.num)-predict(m1,dat) )[-train ]^2)


m1 <- gam(log(weight.num) ~ s(hp, 3) + s(attack, 3) + s(defense, 3) + s(spattack, 3) + s(spdefense, 3) + s(speed, 3), data = dat.train)
pred.m1 <- predict(m1, dat.test)
mean((pred.m1-log(dat.test$weight.num))^2)


m1 <- lm(log(weight.num) ~ bs(hp + attack + defense + spattack + spdefense + speed, df = 3), data = dat.train)
pred.m1 <- predict(m1, dat.test)
mean((pred.m1-log(dat.test$weight.num))^2)

#####################

require(gam)
require(glmnet)
require(nnet)

dat <- read.csv("pokemon.csv", stringsAsFactors = FALSE)

weight.num <- gsub(" lbs.", "", dat$weight)

dat$weight.num <- as.numeric(weight.num)

dat$height.num <- sapply(strsplit(as.character(dat$height), "'|\""),
                         function(x){12 * as.numeric(x[1]) + as.numeric(x[2])})

#####################
# Model 1

# GAM
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 10, ncol = folds)
vec <- 1:1061

for (j in 1:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m1 <- gam(log(weight.num) ~ s(hp, j) + s(attack, j) + s(defense, j) + s(spattack, j) + s(spdefense, j) + s(speed, j), data = dat.train)
    pred.m1 <- predict(m1, dat.test)
    CVmse[j, i] <- mean((pred.m1-test.y)^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Value of s with min CV test MSE
which.min(rowMeans(CVmse)) # Minimum CV test MSE

##########

# Spline
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 8, ncol = folds)
vec <- 1:1061

for (j in 3:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m1 <- lm(log(weight.num) ~ bs(hp + attack + defense + spattack + spdefense + speed, df = j), data = dat.train)
    pred.m1 <- predict(m1, dat.test)
    CVmse[j-2, i] <- mean((pred.m1-log(dat.test$weight.num))^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Minimum CV test MSE
which.min(rowMeans(CVmse)) + 2 # Value of df with min CV test MSE

##########

# Lasso
n <- 1061
folds <- 5
foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID)
for (i in 1:folds) {
  train <- dat[which(foldID != i),]
  test <- dat[which(foldID == i),]
}

train.x <- model.matrix(log(weight.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                          hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                          attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                          spattack*speed+spdefense*speed, train)[,-1]
train.y <- log(train$weight.num)
test.x <- model.matrix(log(weight.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                         hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                         attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                         spattack*speed+spdefense*speed, test)[,-1]
test.y <- log(test$weight.num)


x <- model.matrix(log(weight.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                    hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                    attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                    spattack*speed+spdefense*speed, dat)[,-1]
y <- log(dat$weight.num)


grid <- 10^seq(10, -2, length=100)
lasso.mod <- glmnet(train.x, train.y, alpha = 1, lambda = grid)

set.seed(1)
cv.out = cv.glmnet(train.x, train.y, alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

lasso.pred <- predict(lasso.mod, s = bestlam, newx = test.x)
mean(lasso.pred - test.y)^2 # .000442

out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type='coefficients', s=bestlam)[1:22,]
lasso.coef
lasso.coef[lasso.coef!=0]
length(lasso.coef[lasso.coef!=0])

##########

#Random Forest
require(randomForest)

train = sample(1:dim(College)[1], dim(College)[1] / 2)
test <- -train

dat.train = dat[train, ]
dat.test = dat[test, ]

m1 <- randomForest(log(weight.num)~hp+defense+attack+speed+spattack+spdefense, data = dat.train,importance=TRUE)

pred.rf = predict(m1, dat.test)

mean((pred.rf - log(dat.test$weight.num))^2)

m1$importance

m1

##########

#KNN Regression
require(caret)

set.seed(1)

folds = 5
maxK = 20

x = dat$hp
y = dat$weight.num
foldID = rep(1:folds, length.out = n)
foldID = sample(foldID) # permuting our set

kMSE = rep(0,maxK)

for(k in 1:maxK){
  
  CVmse = rep(0,folds)
  
  for(i in 1:folds){
    
    train <- dat[which(foldID != i),]
    test <- dat[which(foldID == i),]
    
    train.x <- model.matrix(log(weight.num)~hp + attack + defense + spattack + spdefense + speed, train)[,-1]
    train.y <- log(train$weight.num)
    test.x <- model.matrix(log(weight.num)~hp + attack + defense + spattack + spdefense + speed, test)[,-1]
    test.y <- log(test$weight.num)
    
    tempknn = knnreg(train.x, train.y, k = k)
    CVmse[i] = mean((predict(tempknn,test.x) - test.y)^2)
    
  }
  
  kMSE[k] = mean(CVmse)
  
}

testMSE = mean(CVmse)
print(testMSE)

#####################
# Model 2

# GAM
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 10, ncol = folds)
vec <- 1:1061

for (j in 1:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m2 <- gam(log(height.num) ~ s(hp, j) + s(attack, j) + s(defense, j) + s(spattack, j) + s(spdefense, j) + s(speed, j), data = dat.train)
    pred.m2 <- predict(m2, dat.test)
    CVmse[j, i] <- mean((pred.m2-log(dat.test$height.num))^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Value of s with min CV test MSE
which.min(rowMeans(CVmse)) # Minimum CV test MSE

##########

# Spline
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 8, ncol = folds)
vec <- 1:1061

for (j in 3:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m2 <- lm(log(height.num) ~ bs(hp + attack + defense + spattack + spdefense + speed, df = j), data = dat.train)
    pred.m2 <- predict(m2, dat.test)
    CVmse[j-2, i] <- mean((pred.m2-log(dat.test$height.num))^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Minimum CV test MSE
which.min(rowMeans(CVmse)) + 2 # Value of df with min CV test MSE

##########

# Lasso
n <- 1061
folds <- 5
foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID)
for (i in 1:folds) {
  train <- dat[which(foldID != i),]
  test <- dat[which(foldID == i),]
}

train.x <- model.matrix(log(height.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                          hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                          attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                          spattack*speed+spdefense*speed, train)[,-1]
train.y <- log(train$height.num)
test.x <- model.matrix(log(height.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                         hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                         attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                         spattack*speed+spdefense*speed, test)[,-1]
test.y <- log(test$height.num)


x <- model.matrix(log(height.num) ~ hp + attack + defense + spattack + spdefense + speed +hp*attack+hp*defense+
                    hp*spattack+hp*spdefense+hp*speed+attack*defense+attack*spattack+attack*spdefense+
                    attack*speed+defense*spattack+defense*spdefense+defense*speed+spattack*spdefense+
                    spattack*speed+spdefense*speed, dat)[,-1]
y <- log(dat$height.num)


grid <- 10^seq(10, -2, length=100)
lasso.mod <- glmnet(train.x, train.y, alpha = 1, lambda = grid)

set.seed(1)
cv.out = cv.glmnet(train.x, train.y, alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

lasso.pred <- predict(lasso.mod, s = bestlam, newx = test.x)
mean(lasso.pred - test.y)^2 # .0017

out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type='coefficients', s=bestlam)[1:22,]
lasso.coef
lasso.coef[lasso.coef!=0]
length(lasso.coef[lasso.coef!=0])

##########

#Random Forest
train = sample(1:dim(College)[1], dim(College)[1] / 2)
test <- -train

dat.train = dat[train, ]
dat.test = dat[test, ]

m1 <- randomForest(log(height.num)~hp+defense+attack+speed+spattack+spdefense, data = dat.train,importance=TRUE)

pred.rf = predict(m1, dat.test)

mean((pred.rf - log(dat.test$height.num))^2)

##########

#KNN Regression
set.seed(1)

folds = 5
maxK = 20

foldID = rep(1:folds, length.out = n)
foldID = sample(foldID) # permuting our set

kMSE = rep(0,maxK)

for(k in 1:maxK){
  
  CVmse = rep(0,folds)
  
  for(i in 1:folds){
    
    train <- dat[which(foldID != i),]
    test <- dat[which(foldID == i),]
    
    train.x <- model.matrix(log(height.num)~hp + attack + defense + spattack + spdefense + speed, train)[,-1]
    train.y <- log(train$height.num)
    test.x <- model.matrix(log(height.num)~hp + attack + defense + spattack + spdefense + speed, test)[,-1]
    test.y <- log(test$height.num)
    
    tempknn = knnreg(train.x, train.y, k = k)
    CVmse[i] = mean((predict(tempknn,test.x) - test.y)^2)
    
  }
  
  kMSE[k] = mean(CVmse)
  
}

testMSE = mean(CVmse)
print(testMSE)

#####################
# Model 3

# GAM
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 10, ncol = folds)
vec <- 1:1061

for (j in 1:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m3 <- gam(log(total) ~ s(log(weight.num), j) + s(log(height.num), j), data = dat)
    pred.m3 <- predict(m3, dat.test)
    CVmse[j, i] <- mean((pred.m3-log(dat.test$total))^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Value of s with min CV test MSE
which.min(rowMeans(CVmse)) # Minimum CV test MSE

############

# Spline
n <- 1061
folds <- 5

foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID) # permuting our set

CVmse <- matrix(nrow = 8, ncol = folds)
vec <- 1:1061

for (j in 3:10) {
  for(i in 1:folds){
    train <- vec[which(foldID != i)]
    test <- vec[which(foldID == i)]
    
    dat.train <- dat[train, ]
    dat.test <- dat[test, ]
    
    m3 <- lm(log(total) ~ bs(log(weight.num) + log(height.num), df = j), data = dat.train)
    pred.m3 <- predict(m3, dat.test)
    CVmse[j-2, i] <- mean((pred.m3-log(dat.test$total))^2)
  }
}

rowMeans(CVmse) # Get test MSE (average) for each k-fold CV
min(rowMeans(CVmse)) # Minimum CV test MSE
which.min(rowMeans(CVmse)) + 2 # Value of df with min CV test MSE

############

# Lasso
n <- 1061
folds <- 5
foldID <- rep(1:folds, length.out = n)
foldID <- sample(foldID)
for (i in 1:folds) {
  train <- dat[which(foldID != i),]
  test <- dat[which(foldID == i),]
}

train.x <- model.matrix(log(total) ~ log(weight.num) + log(height.num) + log(weight.num*height.num), train)[,-1]
train.y <- log(train$total)
test.x <- model.matrix(log(total) ~ log(weight.num) + log(height.num) + log(weight.num*height.num), test)[,-1]
test.y <- log(test$total)


x <- model.matrix(log(total) ~ log(weight.num) + log(height.num) + log(weight.num*height.num), dat)[,-1]
y <- log(dat$total)


grid <- 10^seq(10, -2, length=100)
lasso.mod <- glmnet(train.x, train.y, alpha = 1, lambda = grid)

set.seed(1)
cv.out = cv.glmnet(train.x, train.y, alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

lasso.pred <- predict(lasso.mod, s = bestlam, newx = test.x)
mean(lasso.pred - test.y)^2 # .00015

out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type='coefficients', s=bestlam)[1:22,]
lasso.coef
lasso.coef[lasso.coef!=0]
length(lasso.coef[lasso.coef!=0])

##########

#Random Forest
train = sample(1:dim(College)[1], dim(College)[1] / 2)
test <- -train

dat.train = dat[train, ]
dat.test = dat[test, ]

m1 <- randomForest(log(total)~log(weight.num)+log(height.num), data = dat.train,importance=TRUE)

pred.rf = predict(m1, dat.test)

mean((pred.rf - log(dat.test$weight.num))^2)


############

#KNN Regression
set.seed(1)

folds = 5
maxK = 20


foldID = rep(1:folds, length.out = n)
foldID = sample(foldID) # permuting our set

kMSE = rep(0,maxK)

for(k in 1:maxK){
  
  CVmse = rep(0,folds)
  
  for(i in 1:folds){
    
    train <- dat[which(foldID != i),]
    test <- dat[which(foldID == i),]
    
    train.x <- model.matrix(log(total) ~ log(weight.num) + log(height.num), train)[,-1]
    train.y <- log(train$total)
    test.x <- model.matrix(log(total) ~ log(weight.num) + log(height.num), test)[,-1]
    test.y <- log(test$total)
    
    tempknn = knnreg(train.x, train.y, k = k)
    CVmse[i] = mean((predict(tempknn,test.x) - test.y)^2)
    
  }
  
  kMSE[k] = mean(CVmse)
  
}

testMSE = mean(CVmse)
print(testMSE)

#####################
# Model 4: Type as response (Weight and Height as predictors)

dat$type1 <- as.factor(dat$type1) # Change type to a factor
dat$type1.norm <- relevel(dat$type1, ref = "Normal")

m1 <- multinom(type1.norm ~ weight.num + height.num + weight.num*height.num, data = dat)
mean(m1$residuals^2) #.0509

m2 <- multinom(type1.norm ~ weight.num + height.num, data = dat)
mean(m2$residuals^2) #.0509

m3 <- multinom(type1.norm ~ weight.num, data = dat)
mean(m2$residuals^2) #.0511

m4 <- multinom(type1.norm ~ height.num, data = dat)
mean(m3$residuals^2) # .0512

z <- summary(m1)$coefficients/summary(m1)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p







