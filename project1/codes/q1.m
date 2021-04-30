%% Question 1

% Clear variables and close figures
clear;
close all;

load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

%% Try different number of hidden units
% Choose network structure
%nHidden = [10];
nHiddens = [5, 10, 20, 50, 100];
maxIter = 100000;
stepSize = 1e-3;

for nHidden = nHiddens
    fprintf("######\nnHidden = %d\n", nHidden);
    
    % Count number of parameters and initialize weights 'w'
    nParams = d*nHidden(1);
    for h = 2:length(nHidden)
        nParams = nParams+nHidden(h-1)*nHidden(h);
    end
    nParams = nParams+nHidden(end)*nLabels;
    w = randn(nParams,1);

    % Train with stochastic gradient
    funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
    tic; % time
    
    for iter = 1:maxIter
        if mod(iter-1,round(maxIter/5)) == 0
            yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
            fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        end
    
        i = ceil(rand*n);
        [f,g] = funObj(w,i);
        w = w - stepSize*g;
    end
   
    toc; % time
    % Evaluate test error
    yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
    fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
end

%% Multiple hidden layers
nHidden = [10, 5];
maxIter = 100000;
stepSize = 1e-3;

nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
fprintf("######\nnHidden = ");
disp(nHidden);
tic; % time

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/5)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

toc; % time
% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
