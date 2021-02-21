close all;
clc;
clear all;

load('trainedbestmodels.mat')


%import test data 
MainData1 = readtable('EEG_test.csv');

MainData = table2array(MainData1); %convert table to array

Xtest = MainData (:, 2:15); %X feature test 
Ytest = MainData (:,16); %Y target test 

%% Accuracies for SVM and MLP models on test set 


%changed labels to 1 2 , dummyvar only accepts positive numbers 
yPred = netfinal(Xtest');
yPredidx = vec2ind(yPred);
yPredidx(yPredidx ==2)=0; %transforming back to 0,1 to evaluate target results 

%Accuracy calculations for MLP on full test set 
accuracymlp = sum(yPredidx' == Ytest)/numel(yPredidx');
fprintf("The Classification Accuracy on the Test Set for MLP is : %.2f%%\n", accuracymlp*100);

%Accuracy calculations for SVM on full test set
prediction_test = str2double(predict(svmmdl, Xtest));
accuracysvm = sum(prediction_test == Ytest)/length(prediction_test);
fprintf("The Classification Accuracy on the Test Set for SVM is : %.2f%%\n", accuracysvm*100);

%% Confusion Matrix for MLP

% Transpose target to match with prediction
figure(1);
plotconfusion(yPredidx,Ytest','EEG Multilayer Perceptron');


%% Confusion Matrix for SVM

% Transpose target and prediction to match 
figure (2);
plotconfusion(prediction_test',Ytest','EEG Support Vector Machine');



        
    

