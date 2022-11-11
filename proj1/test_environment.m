myDir = "./masked_images/";
myFiles = dir(fullfile(myDir));


areas = [];
distances = [];

for i = 1:length(myFiles)
    baseFileName = myFiles(i).name;
    if contains(baseFileName, "jpg")
        % Loading the images
        name_split = split(baseFileName,".");
        distance_string = name_split{1}

    end
end
