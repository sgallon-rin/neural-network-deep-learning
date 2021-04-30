function [y] = Conv_Predict(w1, w2, X, kernel_size, nHidden, nLabels)

X = X(:,2:257);
[nInstances,nVars] = size(X);
height = round(sqrt(nVars));
width = nVars / height;

% Form Weights
nkernels = nHidden(1);
inputWeights = reshape(w1, nkernels, kernel_size^2);
nHidden(1) = nVars * nkernels;
offset = 0;
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w2(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h);
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    image = reshape(X(i,:), height, width);
    for j = 1:nkernels % convolution
        kernel{j} = reshape(inputWeights(j,:), kernel_size, kernel_size);
        featmap{j} = conv2(image, kernel{j}, 'same');
        ip{1}((j-1)*nVars+1:j*nVars) = reshape(featmap{j}, 1, nVars); % reshape as 1d array
    end
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end} * outputWeights;
end

[v,y] = max(y,[],2);
end
