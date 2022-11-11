
% threshold = 0
numofClusters = 3;

% Training GMM 
testDir = "./test_images/";
trainDir = "./train_images/";

myFiles = dir(fullfile(trainDir));

for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    if contains(baseFileName, "jpg")
         % Loading the images
        fullFileName = fullfile(trainDir, baseFileName);
        A = imread(fullFileName);

        % Convert RGB image to chosen color space
        I = rgb2hsv(A);
        
        %  Hue : Define thresholds for channel 1 based on histogram settings
        channel1Min = 0.674;
        channel1Max = 0.066;
        
        %  Saturation: Define thresholds for channel 2 based on histogram settings
        channel2Min = 0.171;
        channel2Max = 0.813;
        
        % Value: Define thresholds for channel 3 based on histogram settings
        channel3Min = 0.587;
        channel3Max = 1.000;
        
        % Create mask based on chosen histogram thresholds
        sliderBW = ( (I(:,:,1) >= channel1Min) | (I(:,:,1) <= channel1Max) ) & ...
            (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
            (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
        BW = sliderBW;
        
        % Initialize output masked image based on input image.
        maskedRGBImage = A;
        
        % Set background pixels where BW is false to zero
        % Image is 640x480x3
        maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
        allImages{k} = maskedRGBImage;
        R = maskedRGBImage(:,:,1); 
        G = maskedRGBImage(:,:,2); 
        B = maskedRGBImage(:,:,3);
        %         I = maskedRGBImage;
        R_val = R(R > 0);
        G_val = G(G > 0);
        B_val = B(B > 0);
        
        orange_pixels = [R_val, G_val, B_val];
        size(orange_pixels);
        pixels = cat(1, orange_pixels, orange_pixels);
    end
end

R = pixels(:,1); 
G = pixels(:,2); 
B = pixels(:,3);

trainingModel = train_gmm;
model = trainingModel.getModel(allImages, orange_pixels);

plotter = plot_gmm
plotter.plot_ellipsoid(model)


% Testing the model

testModel = test_gmm;

% trainFiles = dir(fullfile(trainDir));
% images = testModel.test_model(trainFiles, trainDir, numofClusters, model, "masked_train_images/");

% for i=1:length(images)
%     imshow(images{i});
% end

testFiles = dir(fullfile(testDir));
% create the masked images for measureDepth to test on
images = testModel.test_model(testFiles, testDir, numofClusters, model, "masked_test_images/");
% 
% 
measuringDepth = measureDepth;

% TODO: FOR SOME REASON THE IMAGES CELL ARRAY CONTAINS TWO EMPTY CELLS,
% WHICH IS CURRENTLY ACCOUNTED FOR IN MEASURE_DEPTH.  THIS NEEDS TO BE
% LOOKED INTO

depth = measuringDepth.getDepth(images);
