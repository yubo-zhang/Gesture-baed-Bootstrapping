% this file is used to generate different augmentation to use in the future

%% random crop three times to fix size()

% rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/images/';
% gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/masks/';
% 
% file_id = dir(strcat(rgb_fd,'*png'));
% len = length(file_id);
% num = 1;
% mkdir('./videos/cali_test/GOPR0508/input_result/try_aug/4')
% mkdir('./videos/cali_test/GOPR0508/input_result/try_aug/4/images');
% mkdir('./videos/cali_test/GOPR0508/input_result/try_aug/4/masks');
% st_row = 380;
% st_col = 1030;
% for i = 1:len
%     im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
%     gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
%     [row,col,~] = size(im);
%     row_list = randi([1,row-st_row+1],1,10);
%     col_list = randi([1,col-st_col+1],1,10);
%     for j = 1:3
%         new_im = im(row_list(j):row_list(j)+st_row-1,col_list(j):col_list(j)+st_col-1,:);
%         new_gt_im = gt_im(row_list(j):row_list(j)+st_row-1,col_list(j):col_list(j)+st_col-1);
%         imwrite(new_im, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/images/',int2str(num),'.png'));
%         imwrite(new_gt_im, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/masks/',int2str(num),'.png'));
%         num=num+1;
%     end
% end


%fd = {'GOPR0620';'GOPR0633';'GOPR0634';'GOPR0638';'GOPR0641';'GOPR0660';'GOPR0662'};
fd = {'GOPR0673';'GOPR0676';'GOPR0683'};
for f = 1:length(fd)
%% random crop 6 times to relative small size (keep the ratio: 380*1030)and then scale to the needed size

rgb_fd = strcat('./videos/cali_test/',fd{f,1},'/input_result/RGBimages1/');
gt_fd = strcat('./videos/cali_test/',fd{f,1},'/input_result/Masks1/');
mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages'));
mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks'));

file_id = dir(strcat(rgb_fd,'*png'));
num = 1;
len = length(file_id);
for i = 1: len-1
    im = imread(strcat(rgb_fd,int2str(i),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(i),'.png'));
    imwrite(im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/',int2str(num),'.png'));
    imwrite(gt_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/',int2str(num),'.png'));
    num=num+1;
end

num_to_aug = 50;
%mkdir('./videos/cali_test/GOPR0508/input_result/crop_scale');
%mkdir('./videos/cali_test/GOPR0508/input_result/crop_scale_gt');
st_row = [285];
st_col = [772];
im_list = randi([1,len-1],1,num_to_aug);
for i = 1:num_to_aug
    im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
    %new_im = zeros(380,1030,3);
    [row,col,~] = size(im);
    for k = 1:1
        row_list = randi([1,row-st_row(k)+1],1,2);
        col_list = randi([1,col-st_col(k)+1],1,2);
        for j = 1:2
            new_im1 = im(row_list(j):row_list(j)+st_row(k)-1,col_list(j):col_list(j)+st_col(k)-1,:);
            new_gt_im1 = gt_im(row_list(j):row_list(j)+st_row(k)-1,col_list(j):col_list(j)+st_col(k)-1);
            new_im = imresize(new_im1,[380,1030]);
            new_gt_im = imresize(new_gt_im1,[380,1030]);
            new_gt_im(new_gt_im>0.5)=1;
            new_gt_im(new_gt_im<=0.5)=0;
            imwrite(new_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/',int2str(num),'.png'));
            imwrite(new_gt_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/',int2str(num),'.png'));
            num=num+1;
        end
    end
end


% scale image to smaller size(0.6,0.7,0.8,0.9) and translation randomly for 2 times

% rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/images/';
% gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/masks/';

file_id = dir(strcat(rgb_fd,'*png'));
len = length(file_id);
%num = 1;
%mkdir('./videos/cali_test/GOPR0508/input_result/scale');
%mkdir('./videos/cali_test/GOPR0508/input_result/scale_gt');

st_row = 380;
st_col = 1030;
im_list = randi([1,len-1],1,num_to_aug);

scale = {0.7;0.8};
%num = 1;
for i = 1:num_to_aug
    im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
    for k = 1:length(scale)
        new_im = imresize(im,scale{k,1});
        new_gt = imresize(gt_im,scale{k,1});
        new_gt(new_gt>0.5) = 1;
        new_gt(new_gt<=0.5) = 0;
        [row,col,~] = size(new_im);
        row_list = randi([1,st_row-row+1],1,1);
        col_list = randi([1,st_col-col+1],1,1);
         j=1;
          background_im = uint8(zeros(st_row,st_col,3));
            background_gt = uint8(zeros(st_row,st_col));
            background_im(row_list(j):row_list(j)+row-1,col_list(j):col_list(j)+col-1,:) = uint8(new_im);
            background_gt(row_list(j):row_list(j)+row-1,col_list(j):col_list(j)+col-1) = uint8(new_gt(:,:));
            imwrite(background_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/',int2str(num),'.png'));
            imwrite(background_gt, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/',int2str(num),'.png'));
            num=num+1;
    end
end



%% Rotation image +- 45 degree, every 15 degree save one
%rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/images/';
%gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/masks/';

file_id = dir(strcat(rgb_fd,'*png'));
len = length(file_id);
im_list = randi([round((len-1)/4),round((len-1)*3/4)],1,num_to_aug);
%num = 1;
%mkdir('./videos/cali_test/GOPR0508/input_result/rotate');
%mkdir('./videos/cali_test/GOPR0508/input_result/rotate_gt');
min_angle =-45;
max_angle = 45;
for i = 1:num_to_aug
    im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
    for k = min_angle:30:max_angle
        new_im = imrotate(im,k,'bilinear','crop');
        new_gt = imrotate(gt_im,k,'bilinear','crop');
        new_gt(new_gt>0.5)=1;
        new_gt(new_gt<=0.5) = 0;
        imwrite(new_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/',int2str(num),'.png'));
        imwrite(new_gt, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/',int2str(num),'.png'));   
        num=num+1;
    end
end

        
       

%% random translation for 2 times 

%rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/images/';
%gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/masks/';

file_id = dir(strcat(rgb_fd,'*png'));
len = length(file_id);
im_list = randi([1,len-1],1,num_to_aug);
%num = 1;
%mkdir('./videos/cali_test/GOPR0508/input_result/translation');
%mkdir('./videos/cali_test/GOPR0508/input_result/translation_gt');

for i = 1:num_to_aug
    im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
    [row,col,~] = size(im);
    row_list = double(randi([-row/5,row/5],1,6));
    col_list = double(randi([-col/2,col/2],1,6));
    for j = 1:2
        new_im = imtranslate(im,[col_list(j),row_list(j)]);
        new_im_gt = imtranslate(gt_im,[col_list(j),row_list(j)]);
        imwrite(new_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/',int2str(num),'.png'));
        imwrite(new_gt_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/',int2str(num),'.png'));
            
        num=num+1;
    end
end



%% Brightness

rgb_fd = strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages/');
gt_fd =  strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks/');
mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages1'));
mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks1'));
% fd ={'GOPR0578'};
% f =1;
% rgb_fd = strcat('./videos/cali_test/',fd{f,1},'/input_result/RGBimages1/');
% gt_fd =  strcat('./videos/cali_test/',fd{f,1},'/input_result/Masks1/');
% mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages_br'));
% mkdir(strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks_br'));
file_id = dir(strcat(rgb_fd,'*png'));
len = length(file_id);
num = 1;

%mkdir('./videos/cali_test/GOPR0508/input_result/brightness');
%mkdir('./videos/cali_test/GOPR0508/input_result/brightness_gt');
for i = 1:len-1
    im = imread(strcat(rgb_fd,int2str(i),'.png'));
    gt_im = imread(strcat(gt_fd,int2str(i),'.png'));
    im_hsv = rgb2hsv(im);
    std = im_hsv(:,:,3);
    scale = sum(sum(std,1),2)/1030/380;
    
    for j = 0.3:0.1:0.7
        modify = std;
        modify = modify/scale*j;
        im_hsv(:,:,3) = modify;
        rgb = hsv2rgb(im_hsv);
        imwrite(rgb, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/RGBimages1/',int2str(num),'.png'));
        imwrite(gt_im, strcat('./videos/cali_test/',fd{f,1},'/input_result/augmentation/Masks1/',int2str(num),'.png'));
        num=num+1;
    end
end


end



% %% contrast
% 
% rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/images/';
% gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/3/masks/';
% 
% file_id = dir(strcat(rgb_fd,'*png'));
% len = length(file_id);
% %num = 1;
% %mkdir('./videos/cali_test/GOPR0508/input_result/contrast');
% %mkdir('./videos/cali_test/GOPR0508/input_result/contrast_gt');
% for i = 1:len-1
%     im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
%     gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
%     im_hsv = rgb2hsv(im);
%     std = im_hsv(:,:,2);
%     modify = std;
%     for j = 0.5:0.1:1.8
%         modify = std;
%         modify = modify*j;
%         im_hsv(:,:,2) = modify;
%         rgb = hsv2rgb(im_hsv);
%         imwrite(rgb, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/images/',int2str(num),'.png'));
%         imwrite(gt_im, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/masks/',int2str(num),'.png'));
%         num=num+1;
%     end
% end





%%% ZCA whitening(PCA) to learn dominant features
% 
% rgb_fd = './videos/cali_test/GOPR0508/input_result/rand_crop/';
% gt_fd = './videos/cali_test/GOPR0508/input_result/rand_crop_gt/';
% 
% file_id = dir(strcat(rgb_fd,'*png'));
% len = length(file_id);
% %num = 1;
% mkdir('./videos/cali_test/GOPR0508/input_result/zca');
% mkdir('./videos/cali_test/GOPR0508/input_result/zca_gt');
% 
% cal_size = 100;
% epsilon = 2.2204e-16;
% 
% for i = 1:len-1
%     im = double(imread(strcat(rgb_fd,int2str(im_list(i)),'.png')));
%     gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
%     [row,col,dim] = size(im);
%     new_im = zeros(row,col,dim);
%     for row_index = 1:95:row
%         for col_index = 1:103:col
%             x = reshape(im(row_index:row_index+94,col_index:col_index+102,:),[],3);
%             avg = mean(x, 1);
%             sigma = x * x' / size(x, 2);
%             x = x - repmat(avg, size(x, 1), 1);
%             [U,S,V] = svd(sigma);
%             zac= U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
%             patch = reshape(zac,[95,103,3]);
%             new_im(row_index:row_index+94,col_index:col_index+102,:) = patch;
%         end
%     end
%     num = num+1;
%     imwrite(new_im, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/2/images/',int2str(num),'.png'));
%     imwrite(gt_im, strcat('./videos/cali_test/GOPR0508/input_result/try_aug/2/masks/',int2str(num),'.png'));
% end
% 
% %% add background
%  rgb_fd = './videos/cali_test/GOPR0508/input_result/try_aug/2/images/';
%  gt_fd = './videos/cali_test/GOPR0508/input_result/try_aug/2/masks/';
%  bg_fd = './videos/cali_test/GOPR0508/input_result/background/';
%  bg_file = dir(strcat(bg_fd,'*png'));
%  len1 = length(bg_file);
% file_id = dir(strcat(rgb_fd,'*png'));
% len = length(file_id);
% num = 1;
% mkdir('./videos/cali_test/GOPR0508/input_result/try_aug/4/images');
% mkdir('./videos/cali_test/GOPR0508/input_result/try_aug/4/masks');
% 
% for i = 1:len-1
%      im = imread(strcat(rgb_fd,int2str(im_list(i)),'.png'));
%      gt_im = imread(strcat(gt_fd,int2str(im_list(i)),'.png'));
%      gt  = repmat(gt_im,[1,1,3]);
%      for j = 1:len1-1
%          bg_im = imread(strcat(bg_fd,int2str(j),'.png'));
%          bg_im(gt>0) = im(gt>0);
% %          for r = 1:380
% %              for c = 1:1030
% %                  if gt_im(r,c)>0
% %                      bg_im(r,c,:) = im(r,c,:);
% %                  end
% %              end
% %          end
%          imwrite(bg_im,strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/images/',num2str(num),'.png'));
%          imwrite(gt_im,strcat('./videos/cali_test/GOPR0508/input_result/try_aug/4/masks/',num2str(num),'.png'));
%          num = num+1;
%      end
% end








