%% Question 10

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

%% Use Convolution layer
% hyperparams
maxIter = 100000;
stepSize = 1e-3;

% Choose network structure
nHidden1 = [10]; % first hidden layers' sizes, fully-connected
kernel_size = 5; % convolution filter size
nHidden2 = [1]; % 1: number of conv layers; 2~end: second hidden layers' sizes, fully-connected

fprintf("######\nUse convolution layer, nHidden1 = %d, kernal size = %d, nhidden2 = %d\n", nHidden1, kernel_size, nHidden2);
    
% Count number of parameters and initialize weights 'w'
convParams = kernel_size^2 * nHidden2(1);
w1 = randn(convParams, 1);
if length(nHidden2) > 1
    connectParams = nHidden2(end) * nLabels;
    connectParams = connectParams + nHidden2(1) * d * nHidden2(2);
else
    connectParams = nHidden2(1) * (d-1) * nLabels;
end
for h = 3:length(nHidden2)
    connectParams = connectParams + nHidden2(h-1) * nHidden2(h);
end
w2 = randn(connectParams, 1);

funObj = @(w1,w2,i)ConvNet(w1, w2, X(i, :), yExpanded(i, :), kernel_size, nHidden2, nLabels);
tic; % time
    
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/5)) == 0
        yhat = Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden2, nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f, g1, g2] = funObj(w1, w2, i);
    w1 = w1 - stepSize * g1;
    w2 = w2 - stepSize * g2;
end
    
toc; % time
% Evaluate test error
yhat = Conv_Predict(w1, w2, Xtest, kernel_size, nHidden2, nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
