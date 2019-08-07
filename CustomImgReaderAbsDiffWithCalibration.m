function data = CustomImgReaderAbsDiffWithCalibration(filename)
    img = imread(filename);
    data = imabsdiff(img(1:16,:), img(17:32,:));
end