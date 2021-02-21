
close all;
clc;
clear all;

% Import pre-processed train data 
MainData1 = readtable('EEG_train.csv');
MainData = table2array(MainData1); %convert table to array

% Create X feature and Y target feature variables
Xtrain = MainData (:, 2:15); 
Ytrain = MainData (:,16); 

%% Training best model on full train set for SVM 

labels = {'0', '1'}; % declare class labels 

%Input of best hyperparameters obtained from grid search into model
svmmdl = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'Gaussian',...
         'BoxConstraint', 10,...
          'KernelScale', 100, ...
          'ClassNames',labels, ...
          "Standardize" , false); 
       
%% Train best model on full train set MLP 


% Converting Y target feature to 2 to pass through dummyvar, 2 represents
% eyes open
Ytrain(Ytrain ==0)=2;
ymlp= dummyvar(Ytrain);

% Training best MLP model using feedforwardnet from optimal hyperparameter
% results ouput of bayesopt. Training model on entire train set 
hiddenLayerSize = ones(1, 2) * 65;
netfinal = feedforwardnet(hiddenLayerSize, char('trainlm'));
netfinal.trainParam.lr = 0.00514727497203231; 
netfinal.trainParam.mc = 0.821653988446799; 
netfinal.divideMode = 'none';  

% Apply the function for each layer of network with tansig
for i = 1:2 
    net.layers{i}.transferFcn = char('tansig'); 
end

% Train final model on entire train set, including y dummy target feature
[netfinal , trainingrecord] = train(netfinal, Xtrain', ymlp');


%Save both models to be used in evaluation script 
