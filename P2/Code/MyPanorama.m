dirName = "/Users/riyakumari/Desktop/426/P2/Images/Set1";

myFiles = dir('../Images/Set1');
images = {};

% Reading the images into a cell array
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    if contains(baseFileName, "jpg")
         % Loading the images
        fullFileName = fullfile(dirName, baseFileName);
        A = imread(fullFileName);
        images{k} = A;
    end
end

myPano(images)

function [panorama] = myPano(images)
    % Must load images from ../Images/Input/
    % Must return the finished panorama.

    % Curr image is image 1
%     
%     location = "..\Images\Set1\";
% 
%     loc=dir(location + '*.jpg');
%     num_images=size(loc,1);
%     image_size = zeros(num_images,2);

    descriptors = {};
    img_pts = {};
    
    for i = 3:length(images)
        % Corner detection by converting to grayscale
        curr_img = images{i};
%         imshow(curr_img);
%         hold on;

        curr_img_gs = rgb2gray(curr_img);
%         disp(size(curr_img_gs));
        
        % Store the size of curr_img
        image_size(i,:) = size(curr_img_gs);
        
        % Detect corners 
        curr_img_crnr = cornermetric(curr_img_gs);
       
        % Get best regional corners on these images
        curr_img_mxs = imregionalmax(curr_img_crnr);

        % Run ANMS
        [curr_img_x_best, curr_img_y_best] = ANMS(curr_img, curr_img_mxs, curr_img_crnr);
        img_pts{i-2} = cat(1, curr_img_x_best, curr_img_y_best);
%         
%         To view ANMS results
%         imshow(curr_img)
%         hold on;
%         scatter(curr_img_x_best, curr_img_y_best, 10,'filled');
%         hold off;
        
        % Get feature Descriptors
        descriptors{i-2} = getFeatureDescriptors(curr_img_x_best, curr_img_y_best, curr_img);
        
        
    end
    

%   Feature matching using feature descriptors
    feature_matches1 = {};
    feature_matches2 = {};

    for i=3:length(images)
       
       img1_descriptors = descriptors{i-2}; % descriptor for each image
       img1_pts = img_pts{i-2}; % each image has (x,y) points associated with each descriptor
       
       for j=4:length(images)
            img2_descriptors = descriptors{j-2};
            img2_pts = img_pts{j-2};
            [feature_matches_img1, feature_matches_img2] = getFeatureMatches(img1_descriptors, img2_descriptors, curr_img, img1_pts, img2_pts);
            feature_matches1{i-2, j-2} = feature_matches_img1;
            feature_matches2{i-2, j-2} = feature_matches_img2;
       end
    end
   
    disp(size(transpose(feature_matches1{1,2})))

    showMatchedFeatures(images{3}, images{4}, transpose(feature_matches1{1,2}), transpose(feature_matches2{1,2}), "montage")
    showMatchedFeatures(images{4}, images{5}, transpose(feature_matches1{2,3}), transpose(feature_matches2{2,3}), "montage")
    
    % Use feature_matches{1,2} for feature matching between image1 and image2
    % and feature matches{2,3} for feature matching between image2 and image3
    % feature_matches1 are the x,y points in the first image
    % feature_matches2 are the x,y points in the second image     
    % So if point1 in image 1 matches point2 in image 2 then it'd be like this : feature_matches1{1,2}(:,1) matches feature_matches2{1,2}(:,1)


end
       
function [x_best, y_best] = ANMS(img, regional_maxes, corner_scores)
    n_best = 250;  % fixed number
    n_strong = 0;
    [y_size, x_size, ~] = size(img);

    x_max = [];
    y_max = [];
    for x = 1:x_size
        for y = 1:y_size
            if regional_maxes(y,x) == true
                n_strong = n_strong + 1;
                x_max = [x_max x];
                y_max = [y_max y];  % array concatenation
            end
        end
    end

    
    r = Inf(1, n_strong);
    for i = 1:n_strong
        for j = 1:n_strong
            if corner_scores(y_max(j), x_max(j)) > corner_scores(y_max(i), x_max(i))
                ED = ((x_max(j) - x_max(i))^2) + ((y_max(j) - y_max(i))^2);
                if ED < r(i)
                    r(i) = ED;
                end
            end
        end
    end
    
    x_best = zeros(1, n_best);
    y_best = zeros(1, n_best);

    [~, I] = sort(r, 'descend');
    for i = 1:n_best
        indx = I(i);
        x_best(i) = x_max(indx);
        y_best(i) = y_max(indx);
    end
   
end


% Takes in x and y values of the corners and output vectors
function [descriptors] = getFeatureDescriptors(x_vals, y_vals, img,img1_pts, img2_pts)
    img = rgb2gray(img);
    descriptors = {};

    for i=1:length(x_vals)
        x = y_vals(i); % these are the rows
        y = x_vals(i); % these are the columns
        [rows, cols, numberOfColorChannels] = size(img);
        val = 19;
        startRow = x - val;
        startCol = y - val;
        endRow = x + val;
        endCol = y + val;

%         Could instead try padding the array to fix this issue, right now I'm just skipping these descriptors?
%         if x-val<=0
%            startRow = 1;
%         end
% 
%         if y-val<=0
%             startCol = 1;
%         end
% 
%         if x+val >= rows
%             endRow = rows;
%         end
% 
%         if y+val >= cols 
%             endCol = cols;
%         end

        if x-val<=0 || y-val<=0 || x+val >= rows || y+val >= cols 
            continue;
        end

        % constructing the 40x40 patch
        patch = img(startRow:endRow, startCol:endCol, :);
%         imshow(patch) % uncomment this to see the patches

        % Adding gaussian blur
        kernel = fspecial('gaussian', [7 7], 1.6);
        blurredImage = imfilter(patch, kernel);

        % Downsampling
        downSample = imresize(blurredImage, 0.2, 'nearest'); % downsampling by a factor of 5
        vector = reshape(downSample,[64,1]);
        vector = single(vector);
        mu = mean(vector);
        std_dev = std(vector);
        vector = (vector - mu)/std_dev;
        descriptors{i} = vector;  
        

    end
%     disp(size(descriptors))

end

function [feature_correspondences1, feature_correspondences2] = getFeatureMatches(img1_descriptors, img2_descriptors, img, img1_pts, img2_pts)
    feature_correspondences1 = [];
    feature_correspondences2 = [];
    count = 0;
    
    % looping over each point in img 1
    for i = 1:length(img1_descriptors)        
        pt1 = img1_descriptors{i};
        best_match = Inf;
        second_best_match = Inf;
        best_match_pts = [];
        second_best_match_pts = [];
        

        % looping over each point in img 2
        if size(pt1, 1) == 64
            for j = 1:length(img2_descriptors)
                pt2 = img2_descriptors{j};

                % calculating sum of squared differences
                if size(pt2, 1) == 64
                    sum_diff = sum((pt1-pt2).^2);
                    img1_pt = [img1_pts(1,i), img1_pts(2,i)];
                    img2_pt = [img2_pts(1,j), img2_pts(2,j)];
                    pts = cat(1, img1_pt ,img2_pt); % top row is img1 points and bottom row is img2 points
                    

                    if second_best_match == Inf 
                        second_best_match = sum_diff;
                        second_best_match_pts = pts;
                        
                    elseif best_match == Inf && second_best_match <= sum_diff
                        best_match = second_best_match;
                        best_match_pts = second_best_match_pts;

                        second_best_match = sum_diff;
                        second_best_match_pts = pts;

                    elseif best_match == Inf && second_best_match > sum_diff
                        best_match = sum_diff;
                        best_match_pts = pts;

                    elseif sum_diff < best_match
                        second_best_match = best_match;
                        second_best_match_pts = best_match_pts;
                        best_match = sum_diff;
                        best_match_pts = pts;
                        
                    else
                    end
                    
                end
            end
            ratio = (best_match)/(second_best_match);
            
            if ratio < 0.2 && size(pt1, 1) == 64
                    count = count + 1;
                    points1 = transpose(best_match_pts(1,:)); % points on img1 that correspond with img2
                    points2 = transpose(best_match_pts(2,:)); % points on img2 that correspond with img1

                    feature_correspondences1 = [feature_correspondences1, points1];
                    feature_correspondences2 = [feature_correspondences2, points2];
            end
        end
    end
    
    





% RANSAC FUNCTION DETAILS: points_img*, both 1 and 2, are each Nx2 matrices. 
% Each row in the matrices is a pair of x,y, coordinates
% num_iters is positive int
% threshold is positive number
% Returns homography matrix of only inliers
% Delete the printing if it's annoying
% returns a 3x3 matrix

function robust_homo = ransac_to_est_robust_homography(points_img1, points_img2, num_iterations, threshold)


% add inliers in here later
% these will be used to compute the final homography matrix

little_x_inliers = [];
little_y_inliers = [];
big_x_inliers = [];
big_y_inliers = [];


%--ransac iteration 1 starts here. Will do ransac for 'num_iterations' ---%

for ransac_iteration = 1:num_iterations
    fprintf('ransac iteration = ')
    disp(ransac_iteration);

    %choosing 4 random rows indices
    rows = randi([1 size(points_img1, 1)],4,1);

    %gather all x coordinates from rows specified from first column of IMAGE1
    x = points_img1(rows,1);
    fprintf('x value')
    disp(x);


    %gather all  y coordinates from rows specified in second column of IMAGE1
    y = points_img1(rows, 2);
    fprintf('y value');
    disp(y);

    %gather all X coordinates from rows specified from first column of IMAGE2
    X = points_img2(rows,1);
    fprintf('Big X value');
    disp(X);


    %gather all Y coordinates from rows specified from first column of IMAGE2
    Y = points_img2(rows, 2);
    fprintf('Big Y value');
    disp(Y);

    %apply est_homography function using the 2 x,y pairings
    H = est_homography(X,Y,x,y);
    fprintf('This is returned by estimated homography for the sampling');
    disp(H);


    %x prime and y prime are each vectors of numbers, H is a homography matrix for
    %image1 and image2 sample coordinates

    % applying homography to ALL of image 1's x values, image 1's y values,
    % and using previously genearted H. New H will be used for each
    % iteration of ransac (outer loop)


    [x_prime, y_prime] = apply_homography(H,points_img1(:,1),points_img1(:,2));

    fprintf(" values after applying homographingy of H generated from sample, using points1 and points2")
    disp(x_prime);
    disp(y_prime);

    % Inner for loop:

    % Compare x' and y' values with big X and big Y values
    % Calculate the euclidean distance from points above.
    % If under or = to threshold, it's an inlier. Keep running track of all
    % inliers. Inliers may be double counted. I'm not sure if the est_homography function
    % needs unique coordinates only.


    % applying homography to all of points1 and points2 will always result
    % in 4 coordinates, so loop through 4 coordinates

    for index=1:4
        fprintf('iteration = ');

        big_x = X(index);
        big_y = Y(index);
        prime_x = x_prime(index);
        prime_y = y_prime(index);

        %might want to double check on this formula. Got it from Piazza

        euclidean_distance = sqrt((big_x-prime_x )^2 + (big_y-prime_y)^2);
        fprintf("current euclidean distnace:")
        disp(euclidean_distance);

        % mark coordinates corresponding to row at 'index' for all (x,y) and(X, Y) as inliers.
        % if under threshold, append coordinates to inliers tracker
        % variables. Each index in the variables should correspond, so
        % little_x_inliers(0) and little_y_inliers(0) are corresponding x and y
        % coordinates. Same for big_x_inliers and big_y_inliers

        if euclidean_distance <= threshold

            fprintf(' euclidean distance is <= threshold')

            %append x(index) to little_x_inliers
            little_x_inliers(end+1) = x(index);
            %append y(index) to little_y_inliers(index)
            little_y_inliers(end+1) = y(index);
            %append X(index) to big_x_inliers(index)
            big_x_inliers(end+1) = X(index);
            %append Y(index) to big_y_inliers(index)
            big_y_inliers(end+1) = Y(index);

            % end inner for loop
        end

        %end all ransac iterations
    end

    fprintf('\nBIG X INLIERS ARE:');
    disp(big_x_inliers);

    fprintf('BIG Y INLIERS ARE:');
    disp(big_y_inliers);

    fprintf('LITTLE X INLIERS ARE:');
    disp(little_x_inliers);

    fprintf('LITTLE Y INLIERS ARE:');
    disp(little_y_inliers);

    % recalculate H with the inliers found in the for loop, this gives the
    % homography matrix using only the inliers.

    robust_homo = est_homography(big_x_inliers, big_y_inliers, little_x_inliers, little_y_inliers);

    fprintf('FINAL homography matrix:\n');
    disp(robust_homo);
end

end    




end
