%Image Processing
%Change path name as needed
filepath = "C:\Users\jrcwycy\OneDrive-MichiganMedicine\Desktop\ImageAnalysis\WH_processedimages\03-28-WH\TR4-Circle-1";
fds = fileDatastore(filepath, 'ReadFcn', @imread, 'IncludeSubfolders', true);
fullFileNames = fds.Files;
numFiles = length(fullFileNames);

%%
% % imwrite(BW, 'BW.png');
imwrite(BW, 'BW1.png');
imwrite(BW1, 'BW2.png');
imwrite(BW2, 'BW3.png');
imwrite(BW3, 'BW4.png');
imwrite(BW4, 'BW5.png');
imwrite(BW5, 'BW6.png');
imwrite(BW6, 'BW7.png');
imwrite(BW7, 'BW8.png');
imwrite(BW8, 'BW9.png');
imwrite(BW9, 'BW10.png');
imwrite(BW10, 'BW11.png');
imwrite(BW11, 'BW12.png');
imwrite(BW_filled, 'BW1_filled.png');
% 
% imwrite(maskedImage, 'Mask2.png')
% imwrite(maskedImage1, 'Mask3.png')
% imwrite(maskedImage2, 'Mask4.png')
% imwrite(maskedImage3, 'Mask5.png')
% imwrite(maskedImage4, 'Mask6.png')
% imwrite(maskedImage5, 'Mask7.png')
% imwrite(maskedImage6, 'Mask8.png')
% imwrite(maskedImage7, 'Mask9.png')

%%
%Loop over all images and segment
%Change file names as needed
for k = 1:numFiles
    %fprintf('Now reading file %s\n', fullFileNames{k});
    file = string(fullFileNames(k)); %Get file names in string format
    BW = segmentImage(file); %Segment image
    imwrite(BW, ['BW', num2str(k), '.png']); %Write segmented image to current folder
    %movefile BW* 03-24-Circle_segments\ %Move segmented image to separate folder

    overlayedImage = imoverlay(imread(file), BW, 'yellow'); %Get overlayed image for QC
    imwrite(overlayedImage, ['Mask', num2str(k), '.png']);
    % movefile Mask* 03-24-Circle_overlays\

end

%%   
function BW = segmentImage(filename)

X = imread(filename);

% Create empty mask
BW = false(size(X,1),size(X,2));

row = 1660;
column = 1648;
tolerance = 20;
weightImage = graydiffweight(X, column, row, 'GrayDifferenceCutoff', tolerance); %Geodesic
addedRegion = imsegfmm(weightImage, column, row, 0.01);
% addedRegion = grayconnected(X, row, column, tolerance); %Euclidean
BW = BW | addedRegion;

% % Create masked image.
% maskedImage = X;
% maskedImage(~BW) = 0;

end