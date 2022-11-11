% TODO: ADD COMMENTS
%       VALIDATE THE DEPTH PREDICTIONS FOR ACCURACY
%       IF THEY ARE NOT ACCURATE, CONSIDER MEASURING MORE THAN JUST AREA



classdef measureDepth
    methods
        % this function returns the area of the largest circle in the image
        function area = get_area(obj,input_image)
            image = input_image(:,:,1);
           
            image = image < 100;

            s = regionprops(image,"centroid");
            stats = regionprops('table',image,'Centroid','MajorAxisLength','MinorAxisLength');
                 
            centers = stats.Centroid;

            diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
            radii = diameters/2;
    
            largest_circle_index = find(radii==max(radii));
            
            largest_center = centers(largest_circle_index,:);
            largest_radius = radii(largest_circle_index);
    
            hold on 
            imshow(image)
            viscircles(largest_center,largest_radius);
            hold off

            area = pi * largest_radius^2;
        end
       
        function model = train_depth_model(obj)
            myDir = "./masked_train_images/";
            myFiles = dir(fullfile(myDir));    
            
            areas = [];
            distances = [];

            for i = 1:length(myFiles)
                baseFileName = myFiles(i).name;
                if contains(baseFileName, "jpg")
                    % Loading the images
                    name_split = split(baseFileName,".");
                    distance_string = name_split{1};
                    fullFileName = fullfile(myDir, baseFileName);
                    image = imread(fullFileName);
                    imshow(image)
                    area = obj.get_area(image)
                    areas(end+1) = area;
                    
                    distances(end+1) = str2num(distance_string);

                end
            end
            disp(areas)
            disp(distances)
            model = fit(areas(:),distances(:), 'poly2');




        end


        function depth = getDepth(obj, test_images)
            model = obj.train_depth_model()


            myDir = "./masked_test_images/";
            myFiles = dir(fullfile(myDir));    

            for i = 1:length(myFiles)
                baseFileName = myFiles(i).name;
                if contains(baseFileName, "jpg")
                    % Loading the images
                    name_split = split(baseFileName,".");
                    distance_string = name_split{1};
                    fullFileName = fullfile(myDir, baseFileName);
                    image = imread(fullFileName);
                
                
                    area = obj.get_area(image)
                    predicted_depth = model(area);
    
                    
                    disp("actual depth: ")
                    disp(distance_string)
                    disp("area:")
                    disp(area)
                    disp("predicted depth:")
                    disp(predicted_depth)
                end
            end
            
            depth = predicted_depth;
        end
    end
end

