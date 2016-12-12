%data augmentation with only flip and brightness
file = fopen('../results/index/appearance_index.txt','w');
path = pwd;
home = strsplit(path,'core_files');
home = home{1,1};
    rgb_fd = strcat('../results/gesture/RGBimages/');
    gt_fd = strcat('../results/gesture/Masks/');
    weight_fd = strcat('../results/gesture/Variance/');
    mkdir(strcat('../results/augmentation'));
    mkdir(strcat('../results/augmentation/RGBimages'));
    mkdir(strcat('../results/augmentation/Masks'));
    mkdir(strcat('../results/augmentation/Weights'));
    file_id = dir(strcat(rgb_fd,'*png'));
    num = 1;
    len = length(file_id);
for i = 1: len-1
    im = imread(strcat(rgb_fd,int2str(i),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(i),'.png'));
    weight = dlmread(strcat(weight_fd,int2str(i),'.txt'),',');
    im_hsv = rgb2hsv(im);
    std = im_hsv(:,:,3);
    scale = sum(sum(std,1),2)/1030/380;
    for j = 0.2:0.1:0.7
        modify = std;
        modify = modify/scale*j;
        im_hsv(:,:,3) = modify;
        rgb = hsv2rgb(im_hsv);
        imwrite(rgb, strcat('../results/augmentation/RGBimages/',int2str(num),'.png'));
        imwrite(gt_im, strcat('../results/augmentation/Masks/',int2str(num),'.png'));
        dlmwrite(strcat('../results/augmentation/Weights/',int2str(num),'.txt'),weight);
        fprintf(file,strcat(home, 'results/augmentation/H5_files/', num2str(num),'.h5','\n'));
        num=num+1;
    end
    im_flip= flip(im,2); 
    gt_flip = flip(gt_im,2);
    weight_flip = flip(weight,2);
    im_flip_hsv = rgb2hsv(im_flip);
    std_flip = im_flip_hsv(:,:,3);
    scale_flip = sum(sum(std_flip,1),2)/1030/380;
    for j = 0.2:0.1:0.7 %scale_flip:scale_flip
        modify_flip = std_flip;
        modify_flip = modify_flip/scale_flip*j;
        im_flip_hsv(:,:,3) = modify_flip;
        rgb_flip = hsv2rgb(im_flip_hsv);
        imwrite(rgb_flip, strcat('../results/augmentation/RGBimages/',int2str(num),'.png'));
        imwrite(gt_flip, strcat('../results/augmentation/Masks/',int2str(num),'.png'));
        dlmwrite(strcat('../results/augmentation/Weights/',int2str(num),'.txt'),weight_flip);
        fprintf(file,strcat(home, 'results/augmentation/H5_files/', num2str(num),'.h5','\n'));
        num=num+1;
    end
end 
fclose(file);

 

