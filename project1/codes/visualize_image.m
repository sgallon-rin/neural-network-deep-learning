function visualize_image(X, startAfter, n, pauseSec)

for i = 1:n
    image = X(i + startAfter, 2:257);
    Min = min(image);
    Max = max(image);
    image = (image - Min) / (Max - Min);
    image = reshape(image, 16, 16);
    image = imresize(image, [320, 320]);
    imshow(image);
    pause(pauseSec);
end