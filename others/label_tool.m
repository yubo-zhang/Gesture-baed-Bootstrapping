f = './folder_to_label'; the folder that saves image to label
mkdir(strcat(f,'/masks'));
mkdir(strcat(f,'/RGBimages'));
file_name = strcat(f,'/');
file = dir(strcat(file_name,'*.png'));
index = 70;
for i =1:length(file)
    I = imread(strcat(file_name,int2str(i),'.png'));
    I =imresize(I,[380,1030]);
    h_im = imshow(I);
    h = impoly();
    BW = uint8(createMask(h, h_im));
    imwrite(BW,strcat(f,'/masks/',int2str(index),'.png'));
    imwrite(I,strcat(f,'/RGBimages/',int2str(index),'.png'));
    index=index+1;
    disp(i);
end




