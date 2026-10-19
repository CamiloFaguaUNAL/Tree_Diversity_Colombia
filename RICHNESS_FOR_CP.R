library(randomForest)
library(Boruta)
library(caret)
library(terra)
library(ranger)
library(Metrics)
library(irr)
library(dplyr)
#
setwd("")
#
rm(list = ls(all.names = TRUE)) 
gc()
#(1) Data organization 
TABLA = read.csv("CP_1h_PLOTS.csv", header = TRUE, sep = ",")
names(TABLA)
dim(TABLA)
#
TABLA_1 = TABLA[,c(3,9:78)]
TABLA_1 = TABLA_1[complete.cases(TABLA_1), ]
dim(TABLA_1)
names(TABLA_1)
#
#(2) Removal of non-influential predictors in modeling
set.seed(1)
Var_sel_1 = Boruta(Richness~., data=TABLA_1, doTrace=2)
plot(Var_sel_1)
#
R1 = attStats(Var_sel_1)
R1 = data.frame(rownames(R1),R1)
R1
#
R2 = subset(R1, decision != "Rejected")
R2
#
#write.csv(R2, "Result_boruta_Richness.csv")
rownames(R2)
#
selected_vars = c("Richness", row.names(R2))
TABLA_2 = TABLA_1[,selected_vars]
str(TABLA_2)
#

#
#set.seed(1)
trainIndex = createDataPartition(TABLA_2$Richness, p = .8, 
                                 list = FALSE, 
                                 times = 1)

#(3) Data partion 
head(trainIndex)
Train = TABLA_2[ trainIndex,]
dim(Train)
#write.csv(Train, "Train_Richness.csv")
#
Test = TABLA_2[-trainIndex,]
names(Test)
#write.csv(Test, "Test_Richness.csv")


#(4) Random Forest modeling 
k1=ncol(TABLA_2)-1
#(4.1) Directions for tuning 
control = trainControl(method='repeatedcv', 
                       number=5, 
                       repeats=10) 

mtry_1 = c(2,3,4,5,sqrt(k1),7,8,9)
min.node.size_1 = c(1, 3, 5, 10)
splitrule_1 = c('gini','extratrees')

tunegrid_1 = expand.grid(mtry=mtry_1, 
                         min.node.size=min.node.size_1,
                         splitrule=splitrule_1)

#(4.2) Tuning
rf_1 = train(Richness~., 
             data=Train, 
             method='ranger', 
             metric='rmse', 
             tuneGrid=tunegrid_1, 
             trControl=control)


print(rf_1)
#plot(rf_1)

#(4.3) Selection of the best model-parameters
best_mtry = rf_1$bestTune[[1]]
best_m.n.s = rf_1$bestTune[[3]]
max(rf_1$results$Rsquared, na.rm = T)

#
tunegrid_2 = expand.grid(mtry= best_mtry, 
                         min.node.size=best_m.n.s,
                         splitrule='extratrees')

#(4.4) Run of the best model
rf_2 = train(Richness~., 
             data=Train, 
             method='ranger', 
             metric='RMSE', 
             tuneGrid=tunegrid_2, 
             trControl=control)

rf_2



#(5) Error estimation using the test group
Test_2 = Test[,selected_vars]
names(Test)
TEST_rf_2 = predict(rf_2, Test_2)

TABLON = data.frame(Test[,1], TEST_rf_2, Test$CH)
names(TABLON) = c("Actual", "Predict", "CH")
head(TABLON)
dim(TABLON)
#
TABLON_2 = filter(TABLON, Actual != 0, Predict != 0)
dim(TABLON_2)


RMSE_rf_2 = rmse(actual = TABLON_2[,1], predicted = TABLON_2[,2])
RRSE_rf_2 = rrse(actual = TABLON_2[,1], predicted = TABLON_2[,2])
#
MAE_rf_2 = mae(actual = TABLON_2[,1], predicted = TABLON_2[,2])
RAE_rf_2 = rae(actual = TABLON_2[,1], predicted = TABLON_2[,2])
#
BIAS_rf_2 = bias(actual = TABLON_2[,1], predicted = TABLON_2[,2])
P_BIAS_rf_2 = percent_bias(actual = TABLON_2[,1], predicted = TABLON_2[,2])
#
ABSOLUTE_rf_2 = mean(ae(actual = TABLON_2[,1], predicted = TABLON_2[,2]))
PER_ABSOLUTE_rf_2 = mean(ape(actual = TABLON_2[,1], predicted = TABLON_2[,2]))
#
mase_rf_2 = mase(actual = TABLON_2[,1],predicted = TABLON_2[,2], step_size = 1)
mdae(actual = TABLON_2[,1], predicted = TABLON_2[,2])


RESULTADOS = data.frame(RMSE_rf_2, RRSE_rf_2, MAE_rf_2, RAE_rf_2, 
                        BIAS_rf_2,P_BIAS_rf_2, ABSOLUTE_rf_2, PER_ABSOLUTE_rf_2)


names(RESULTADOS) = c("rmse", "rrse", "mae", "rae", "bias", "p_bias", "mean_abs_error",
                      "mean_%abs_error")

RESULTADOS
