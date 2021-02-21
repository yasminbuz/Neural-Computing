close all;
clc;
clear all;

%Import train pre-processed train data 
MainData1 = readtable('EEG_train.csv');
MainData = table2array(MainData1); %convert table to array

Xtraining = MainData(:, 2:15); %X feature training (ignoring first column (index) 
Ytraining = MainData (:,16); %Y target feature training
 

%% Grid Search for Linear, Polynomial, Gaussian and RBF functions


%Partition training and validation with 30% holdout
cv_val = cvpartition(size(Xtraining, 1), 'HoldOut', 0.3);
idx_val = cv_val.test;

%Create input variables
Xtrain1 = Xtraining(~idx_val, :);
Xval  = Xtraining(idx_val, :);

%Create Y target variables
Ytrain1 = (Ytraining(~idx_val, :));
Yval = (Ytraining(idx_val, :));

%Specify target labels (0 - eyes open and 1- eyes closed) 
labels = {'0', '1'};                
  
%Create the empty matrix for accuracy outputs
val2_acc = zeros(4,4,4);          
training2_acc = zeros(4,4,4);     


% Specifying hyper parameters for grid search, including box constraint
% (C) , kernel scale(G) and kernel function options 
G = [0.1, 1, 10,100];
C = [0.1, 1, 10,100];
kfunction = {'polynomial', 'rbf', 'linear','gaussian'};

%Create for loop to search through all possibilities of hyperparameters
for iter = 1:4 %search through kernel functions
    for i = 1:3 %search through kernel scale
        for j = 1:3 %search through box constraints
        
% Used builtin function SVM model from Matlab with iterating kernel
% functions 
            Mdl = fitcsvm(Xtrain1,Ytrain1,'KernelFunction',kfunction{iter},...
            'BoxConstraint',C(j),...
            'KernelScale',G(i),...
            'ClassNames', labels);
     
        
 % Calculating the training and validation accuracies, changing to double
        
        prediction_train = str2double(predict(Mdl, Xtrain1));
        prediction_val = str2double(predict(Mdl, Xval));
        
 % Calculating training and validation for grid search variations 
        trainingScore = sum(prediction_train==Ytrain1)/length(prediction_train);
        validationScore = sum(prediction_val==Yval)/length(prediction_val);
        
         
 % Output training2 and val2 results matrix
        training2_acc(iter,i,j) = trainingScore;
        val2_acc(iter,i,j) = validationScore;
        
        end
    end
end


           
        

    
       