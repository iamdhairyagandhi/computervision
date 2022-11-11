classdef test_gmm
    methods
        function images = test_model(obj, testFiles, myDir, numofClusters, model, outDir)
            
            mu_i = model(:,1);
            cov_i = model(:,2);
            pi_i = model(:,3);
            step = 1;
            
            for file = 1:length(testFiles)
                baseFileName = testFiles(file).name;
                if contains(baseFileName, "jpg")
                    % Loading the images
                    fullFileName = fullfile(myDir, baseFileName);
                    A = imread(fullFileName);
            
                    img = A;


                    %image=zeros(640,480,3);
                    image=zeros(size(A));
                    for cluster=1:3
                        pixel_count = 0;
            
                        covariance = cov_i{cluster};
                        mean = mu_i{cluster};
                        scale = pi_i{cluster};
            %             
                        for row=1:size(img,1) % looping over pixels of an image
                            for col=1:size(img,2)
                                pixel_count = pixel_count + 1;
                                x = [double(img(row,col,1)); double(img(row,col,2)); double(img(row,col,3))]; % getting a single [r g b] value
                                p = scale*(1/sqrt((2*pi)^3 * det(covariance)))*exp(-0.5*transpose(x-mean)*((covariance)\(x-mean))) * 0.5; % calculating likelihood
                                if p > 3.0740e-100
                                    image(row,col,:) = [0 0 0];
                                else
                                    image(row,col,:) = [255 255 255];
                                end
                            end
                        end
                    end
                   path = strcat("./outputs/gmm/",baseFileName);
                   final_output = imfuse( A, image, 'montage');
                   imwrite(final_output,path);
                   images{step} = image;
                   step = step + 1;
                end
            end
        end


    end
end