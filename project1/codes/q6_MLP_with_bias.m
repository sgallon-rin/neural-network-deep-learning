function [f,g,gb] = q6_MLP_with_bias(w,b,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X); % 5000*257

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1)); % 257*10
offset = nVars*nHidden(1); % =2570

for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % 10*10

% bias
inputBias = b(1:nHidden(1));
offset = nHidden(1);
for h = 2:length(nHidden)
    hiddenBias{h-1} = b(offset+1:offset+nHidden(h));
    offset = offset + nHidden(h);
end
outputBias = b(offset+1:offset+nLabels);

f = 0;
if nargout > 1
    % weight and bias
    gInput = zeros(size(inputWeights)); % 10*10
    gbInput = zeros(size(inputBias));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
       gbHidden{h-1} = zeros(size(hiddenBias{h-1}));
    end
    gOutput = zeros(size(outputWeights)); % 10*10
    gbOutput = zeros(size(outputBias));
end

% Compute Output
for i = 1:nInstances
    % 预分配内存以加快运算速度
    ip = cell(length(nHidden), 1);
    fp = cell(length(nHidden), 1);
    
    ip{1} = X(i,:)*inputWeights + inputBias'; % 1*10
    fp{1} = tanh(ip{1}); % 1*10
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1} + hiddenBias{h-1}';
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end} * outputWeights + outputBias'; % 1*10
    
    relativeErr = yhat - y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr; % 1*10
        
        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c) * fp{end}';
            gbOutput(c) = gbOutput(c) + err(c);
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                % 用来反向传播的中间项，sech双曲正割函数是双曲余弦的倒数
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gHidden{end} = gHidden{end} + fp{end-1}' * backprop(c, :);
                gbHidden{end} = gbHidden{end} + backprop(c, :)';
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
                gbHidden{h} = gbHidden{h} + backprop';
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
            gbInput = gbInput + backprop';
        else
            % length(nHidden) equals 1
            % Input Weights
            %backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            %disp(size(backprop));
            %disp(size(X));
            %gInput = gInput + X' * backprop;
            %gbInput = gbInput + sum(backprop, 2);
            for c = 1:nLabels
                backprop(c,:) = err(c) * (sech(ip{end}) .^ 2 .* outputWeights(:,c)');
                gInput = gInput + X(i,:)' * backprop(c,:);
                gbInput = gbInput + backprop(c, :)';
            end
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    gb = zeros(size(b));
    g(1:nVars*nHidden(1)) = gInput(:);
    gb(1:nHidden(1)) = gbInput(:);
    offset = nVars*nHidden(1);
    offsetb = nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        gb(offsetb+1:offsetb+nHidden(h)) = gbHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
        offsetb = offsetb + nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
    gb(offsetb+1:offsetb+nLabels) = gbOutput(:);
end