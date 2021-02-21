
close all;
clc;
clear all;

% Import training EEG data 
MainData1 = readtable('EEG_train.csv');
MainData = table2array(MainData1); %convert table to array

% Create X feature and Y target feature variables
Xtraining = MainData (:, 2:15); 
Ytraining = MainData (:,16); 

% Transform Y target to class 2, eyes open becomes 2 because dummyvar only accepts positive numbers
% Transformation to dummyvar required to fit model data specification

Ytraining(Ytraining ==0)=2; 
y1= dummyvar(Ytraining);

% Partition Y target feature for validation
cv = cvpartition(size(Ytraining,1), 'Holdout', 1/3); 


%% Bayesian Optimization for best Hyper-Parameters for MLP

% Using Bayesopt built-in function
% Specifying complete range of hyper-parameters for Multi-layer Perceptron
n = optimizableVariable('networkDepth',[1,5],'Type', 'integer');
hn = optimizableVariable('numHiddenNeurons', [1,100], 'Type', 'integer');
lr = optimizableVariable('lr', [0.001 1], 'Transform', 'log');
fcn = optimizableVariable('trainFcn', {'traingda','trainlm', 'traingdm', 'traingdx', 'trainscg'}, 'Type', 'categorical');
m = optimizableVariable('momentum', [0.8 0.95]);
tfcn = optimizableVariable('transferFcn', {'logsig', 'poslin', 'tansig', 'purelin'}, 'Type', 'categorical');

% Note: kfold loss function created at end of script to mark error
% Calling kfoldfunction to X training and target has been transposed to fit
minfn = @(T)kfoldLoss(Xtraining', y1', cv, T.networkDepth, T.numHiddenNeurons,...
    T.lr, T.momentum, T.trainFcn, T.transferFcn);

% Results output of bayesopt assigned to T (the best results of
% optimization)
results = bayesopt(minfn, [n,hn,lr,fcn,m,tfcn],'IsObjectiveDeterministic', false,'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations',200);

% T is recoreded with optimal hyperparameters and outputs used to train  
T = bestPoint(results);

% best model in final model script 

%% kfold loss function to calculate model errors for each epoch

function valerror = kfoldLoss(x, y, cv, networkDepth, numHiddenNeurons, lr, momentum, trainFcn, transferFcn)

% Building neural net, setting network depth size and number of hidden
% layers 
hiddenLayerSize = numHiddenNeurons * ones(1, networkDepth);

% Using built in function for feedforward MLP
net = feedforwardnet(hiddenLayerSize, char(trainFcn)); 

% Number of epochs for bayeopt
net.trainParam.epochs = 50;

% Set stopping criteria to avoid excessive computation
net.trainParam.max_fail = 6; 

% Create separate statement for functions with learning rate parameter
if any(strcmp({'traingda', 'traingdm', 'traingdx'} , char(trainFcn))) 
    net.trainParam.lr = lr;
    
% For functions that includes momentum, parameter is updated
    if ~strcmp('traingda', char(trainFcn)) 
        net.trainParam.mc = momentum; 
    end
end

%Applying the function for each iteration (layer) of network 
for i = 1:networkDepth 
    net.layers{i}.transferFcn = char(transferFcn); 
end

% Train full neural network, with X attribute and Y target feature 
[net, tr] = train(net,x,y); 

% Record output of functions and validation train set errors 
ypred = net(x);
ypredind = vec2ind(ypred); 

% Transform Y prediction to index, to calculate val error 
yind = vec2ind(y);
valerror = sum(ypredind(cv.test) ~= yind(cv.test))/numel(ypredind(cv.test)); %error calculations
end
