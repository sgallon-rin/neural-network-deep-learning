%% Final model

% Clear variables and close figures
clear;
close all

load digits.mat
%load digits_augmented.mat
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
stepSize = 1e-4;

% Choose network structure
kernel_size = 3; % convolution filter size
beta = 0.9; % momentum strength
%nHidden = [4 128 64 32]; % 1: number of conv layers; 2~end: hidden layers' sizes, full-connected
nHidden = [4];

%useXavier = 0;
useXavier = 1;
    
% Count number of parameters and initialize weights 'w'
convParams = kernel_size^2 * nHidden(1);

if useXavier == 1
% 使用Xavier初始化代替随机初始化
w1 = randn(convParams, 1) / (kernel_size * sqrt(nHidden(1)));
inputDims = [nHidden(1)*d, nHidden(2:length(nHidden)), 10];
connectParams = 0;
offset = 0;
for i=1:length(inputDims)-1
    connectParams = connectParams + inputDims(i)*inputDims(i+1);
    offset = offset + inputDims(i)*inputDims(i+1);
end
w2 = zeros(connectParams, 1);
offset = 0;
for i=1:length(inputDims)-1
    w2(offset+1:offset+inputDims(i)*inputDims(i+1),1) = randn(inputDims(i)*inputDims(i+1),1) / sqrt(inputDims(i));
    offset = offset + inputDims(i)*inputDims(i+1);
end
else
w1 = randn(convParams, 1);
if length(nHidden) > 1
    connectParams = nHidden(end) * nLabels;
    connectParams = connectParams + nHidden(1) * d * nHidden(2);
else
    connectParams = nHidden(1) * (d-1) * nLabels;
end
for h = 3:length(nHidden)
    connectParams = connectParams + nHidden(h-1) * nHidden(h);
end
w2 = randn(connectParams, 1);
end

% use momentum
old_w1 = w1; % w_{t-1}
old_w2 = w2;
funObj = @(w1,w2,i)ConvNet(w1, w2, X(i, :), yExpanded(i, :), kernel_size, nHidden, nLabels);

% record dev error
N = 100;
dev_errs = zeros(1, N);
j = 1;

tic; % time

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/N)) == 0
        yhat = Conv_Predict(w1, w2, Xvalid, kernel_size, nHidden, nLabels);
        err = sum(yhat~=yvalid)/t;
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,err);
        dev_errs(j) = err;
        j = j + 1;
    end
    
    i = ceil(rand*n);
    [f, g1, g2] = funObj(w1, w2, i);
    % use momentum
    w1 = w1 - stepSize*g1 + beta*(w1-old_w1);
    w2 = w2 - stepSize*g2 + beta*(w2-old_w2);
    old_w1 = w1;
    old_w2 = w2;
end
    
if useXavier == 1
    fprintf("######\nUse Xavier initialization\n");
else
    fprintf("######\nDo not use Xavier initialization\n");
end
fprintf("Use momentum, momentum strength = %.1f\n", beta);
fprintf("lr = %f\n", stepSize);
fprintf("Use convolution layer, kernal size = %d\n", kernel_size);
fprintf("Number of convolution layers = %d\n", nHidden(1));
if length(nHidden) > 1
    fprintf("nHidden = ");
    disp(nHidden(2:end));
else
    fprintf("no hidden layer\n");
end

toc; % time
% Evaluate test error
yhat = Conv_Predict(w1, w2, Xtest, kernel_size, nHidden, nLabels);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);

%% Plot dev error

iters = linspace(0, maxIter, N+1);
iters = iters(1:N);
figure(1);
plot(iters, dev_errs, "r");
title('Validation error of finall model')
xlabel('iteration');
ylabel('err');
%plot(iters, dev_errs, "b");
hold on;
%legend(["w Xavier", "w/o Xavier"]);
