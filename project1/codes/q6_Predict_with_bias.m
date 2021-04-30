function [y] = q6_Predict_with_bias(w, b, X, nHidden, nLabels)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars * nHidden(1)), nVars, nHidden(1)); 
offset = nVars * nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h); 
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels); 
outputWeights = reshape(outputWeights,nHidden(end),nLabels); 

% bias
inputBias = b(1:nHidden(1));
offset = nHidden(1);
for h = 2:length(nHidden)
    hiddenBias{h-1} = b(offset+1:offset+nHidden(h));
    offset = offset + nHidden(h);
end
outputBias = b(offset+1:offset+nLabels);

% Compute Output
for i = 1:nInstances 
    ip{1} = X(i, :) * inputWeights + inputBias'; 
    fp{1} = tanh(ip{1}); 
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1} + hiddenBias{h-1}'; 
        fp{h} = tanh(ip{h}); 
    end
    y(i,:) = fp{end} * outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
