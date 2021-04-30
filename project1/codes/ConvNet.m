function [f,g1,g2] = ConvNet(w1,w2,X,y,kernel_size,nHidden,nLabels)

X = X(:,2:257); % drop the first column
[nInstances,nVars] = size(X); % 5000*256
height = round(sqrt(nVars));
width = nVars / height;

% Form Weights
nkernels = nHidden(1);
% convolution layer weight
inputWeights = reshape(w1(1:kernel_size^2*nkernels), nkernels, kernel_size^2);
nHidden(1) = nVars * nkernels; % expended length of a figure
offset = 0;
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w2(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset + nHidden(h-1) * nHidden(h);
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
    gInput = zeros(size(w1));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    image = reshape(X(i,:), height, width);
    image = padarray(image, [floor(kernel_size/2), floor(kernel_size/2)]);
    for j = 1:nkernels % convolution
        kernel{j} = reshape(inputWeights(j,:), kernel_size, kernel_size);
        featmap{j} = conv2(image, kernel{j}, 'valid');
        ip{1}((j-1)*nVars+1:j*nVars) = reshape(featmap{j}, 1, nVars); % reshape as 1d array
    end
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end} * outputWeights;
    
    relativeErr = yhat - y(i,:); 
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2 * relativeErr; 

        % Output Weights
        gOutput = gOutput + (err' * fp{end})'; 

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            gHidden{end} = gHidden{end} + repmat(fp{end-1}', 1, nLabels) * backprop;
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* sech(ip{h+1}) .^ 2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end

            % Input Weights
            backprop = (backprop * hiddenWeights{1}') .* sech(ip{1}) .^ 2; % 1 * 768
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        else
            % Input Weights
            backprop = err * (sech(ip{end}) .^ 2 .* outputWeights');
            for j = 1:nkernels
                delta{j} = reshape(backprop((j-1)*nVars+1:j*nVars), height, width);
                grad = conv2(image, imrotate(delta{j}, 180, 'bilinear'), 'valid');
                gInput((j-1)*kernel_size^2+1:j*kernel_size^2) = reshape(grad, kernel_size^2, 1);
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g1 = gInput;
    g2 = zeros(size(w2));
    offset = 0;
    for h = 2:length(nHidden)
        g2(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset + nHidden(h-1) * nHidden(h);
    end
    g2(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
