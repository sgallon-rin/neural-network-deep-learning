function [f,g] = q7_MLP_with_dropout(w,X,y,nHidden,nLabels,p)

[nInstances,nVars] = size(X); % 5000*257

% Form Weights
w = w / (1-p); % need to scale with dropout
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1)); % 257*10
offset = nVars*nHidden(1); % =2570

for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels); % 10*10

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights)); % 10*10
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights)); % 10*10
end

% Compute Output
for i = 1:nInstances
    % 预分配内存以加快运算速度
    ip = cell(length(nHidden), 1);
    fp = cell(length(nHidden), 1);
    
    ip{1} = X(i,:)*inputWeights; % 1*10
    fp{1} = tanh(ip{1}); % 1*10
    fp{1} = fp{1} .* (rand(size(fp{1}))>p); % dropout
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
        fp{h} = fp{h} .* (rand(size(fp{h}))>p); % dropout
    end
    yhat = fp{end}*outputWeights; % 1*10
    
    relativeErr = yhat-y(i,:); % 1*10
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr; % 1*10

        % Output Weights
        %for c = 1:nLabels
        %    gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}';
        %end
        % 改为矩阵运算 循环内是列向量
        gOutput = gOutput + (err' * fp{end})'; % 10*10

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            % 改为矩阵运算 循环内是行向量
            %for c = 1:nLabels
            %    backprop(c,:) = err(c)*(sech(ip{end}).^2.*outputWeights(:,c)'); % 1*10行向量
            %    gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:);
            %end
            backprop = err' .* (sech(ip{end}) .^ 2 .* outputWeights');
            gHidden{end} = gHidden{end} + repmat(fp{end-1}', 1, nLabels) * backprop;
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
        else
            % length(nHidden) equals 1
            % Input Weights
            % 改为矩阵运算
            %for c = 1:nLabels
            %    gInput = gInput + err(c)*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
            %end
            gInput = gInput + err .* X(i,:)' * (sech(ip{end}) .^ 2 .* outputWeights');
        end

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
