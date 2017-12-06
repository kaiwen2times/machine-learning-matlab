close all ; clear all;
load ex4data1.mat;

options.method = 'SVM';
options.numberOfFolds = 5;
options.useDR=1;
options.dim_reduction='SLPP';
options.SLPP_bLDA=0.7;
options.PCARatio=0.9;
[confusionMatrixSVM,accuracySVM] =  classify677_hwk8(X,y,options);

options.method = 'nnet';
options.nnet_hiddenLayerSize = 25;
options.numberOfFolds = 5;
options.useDR=1;
options.dim_reduction='SLPP';
options.SLPP_bLDA=0.7;
options.PCARatio=0.9;
[confusionMatrixNN,accuracyNN] =  classify677_hwk8(X,y,options);
