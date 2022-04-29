clc;
clear;
clear all;

%add current file path and sub file path
addpath(genpath(pwd));
load('Yeast_data.mat');
method.name={'ECC'};
method.param=cell(length(method.name),1);
%% BaseClassifier
% 1. Support Vector Machines
method.base.name='linear_svm';
method.base.param.svmparam='-s 2 -B 1 -q';
[method.param{1}]=feval(['Set',method.name{1},'Parameter'],[]);
method.th.type='SCut';
method.th.param=0.5;
% Y 1/-1 -> 1/0
train_target(train_target == -1) = 0;
test_target(test_target == -1) = 0;
[model,train_time]=ECC_train(train_data,train_target',method);
%
[conf, test_time]=ECC_test(train_data,train_target',test_data,model,method);
%
[pred]=Thresholding(conf,method.th,test_target');



Pre_Labels = pred';
Outputs = conf';
[auc,~,~] = ak_auc_tp_fp_diffrent_ks(Outputs',test_target');
test_target(test_target ==0 ) = -1;
Pre_Labels(Pre_Labels ==0 ) = -1;
HammingLoss=Hamming_loss(Pre_Labels,test_target);

RankingLoss=Ranking_loss(Outputs,test_target);
OneError=One_error(Outputs,test_target);
Coverage=coverage(Outputs,test_target);
Average_Precision=Average_precision(Outputs,test_target);

macrof1 = MacroF1(Pre_Labels,test_target);
microf1 = MicroF1(Pre_Labels,test_target);

