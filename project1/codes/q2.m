%% Question 2

% Clear variables and close figures
clear;
close all

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

% Choose network structure
nHidden = [10];
    
% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;

maxIter = 100000;

%% 1 Train with stochastic gradient, contstant learning rate, try different lr
stepSizes = [1e-3, 1e-2, 1e-1];
for stepSize = stepSizes
    fprintf("######\nSGD with constant lr = %f\n", stepSize)
    w = randn(nParams,1);
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

%% 2 Exponential Decay

w = randn(nParams,1);
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
stepSize = 1e-3;
decayRate = 0.9; % decay rate for exponential decay
fprintf("######\nExponential Decay, lr = %f, decay rate = %f\n", stepSize, decayRate)
tic; % time

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/5)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    stepSize = stepSize * decayRate^(1/maxIter);
    w = w - stepSize*g;
end

toc; % time
% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);

%% 3 Momentum

w = randn(nParams,1);
funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
stepSize = 1e-3;
beta = 0.9; % momentum strength
old_w = w; % w_{t-1}
fprintf("######\nMomentum, lr = %f, momentum strength = %f\n", stepSize, beta)
tic; % time

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/5)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g + beta*(w-old_w);
    old_w = w;
end

toc; % time
% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
