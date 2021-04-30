% data augmentation
tic;
load digits.mat
X = transforms(X);
y = repmat(y, 5, 1);
save digits_augmented.mat X Xtest Xvalid y ytest yvalid
toc;

function [X] = transforms(X)

[nsamples, nfeatures] = size(X);
nrow = round(sqrt(nfeatures));
ncol = round(sqrt(nfeatures));

for i=1:nsamples
    image = X(i,:);
    image = reshape(image, nrow, ncol);
    
    image_pos_15degree = imrotate(image, 15, 'bilinear'); % ��ʱ����ת15��
    image_neg_15degree = imrotate(image, -15, 'bilinear'); % ˳ʱ����ת15��
    image_pos_15degree = imresize(image_pos_15degree, [nrow ncol], 'bilinear'); % ������ת���ͼ��
    image_neg_15degree = imresize(image_neg_15degree, [nrow ncol], 'bilinear'); % ������ת���ͼ��
    image_left_trans = [image(:,3:ncol) zeros(nrow, 2)]; % ����2�����غ��ͼ��
    image_right_trans = [zeros(nrow, 2) image(:,1:ncol-2)]; % ����2�����غ��ͼ��
    
    % ��ͼ����ֱΪ���������ڴ洢
    image_pos_15degree = reshape(image_pos_15degree, 1, nfeatures);
    image_neg_15degree = reshape(image_neg_15degree, 1, nfeatures);
    image_left_trans = reshape(image_left_trans, 1, nfeatures);
    image_right_trans = reshape(image_right_trans, 1, nfeatures);
    
    % �洢�任���ͼ��
    X(nsamples+i,:) = image_pos_15degree;
    X(2*nsamples+i,:) = image_neg_15degree;
    X(3*nsamples+i,:) = image_left_trans;
    X(4*nsamples+i,:) = image_right_trans;
end
end