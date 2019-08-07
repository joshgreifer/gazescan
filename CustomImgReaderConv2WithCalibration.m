function data = CustomImgReaderConv2WithCalibration(filename)
    img = imread(filename);
    img = cast(img,'double') / 256;
    img = img - mean(img(:));
    img = conv2(img(1:16,:), img(17:32,:));
    data = imresize(img, [16,32]);
end