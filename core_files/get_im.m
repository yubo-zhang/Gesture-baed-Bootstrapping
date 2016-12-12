function get_im(video_path)
mkdir('../results/raw_images');
video = VideoReader(video_path);
len = video.NumberOfFrames;

%read middle 180 images
if len <=180
    start = 1;
    ends = len;
else
    start = floor((len-180)/2)+1;
    ends = start+179;
end
file = fopen('../results/raw_images/index.txt','w');
for i=start: ends
    frame = read(video,i);
    imwrite(frame, strcat('../results/raw_images/',int2str(i),'.png'));
    fprintf(file,strcat(int2str(i),'.png'));
    fprintf(file,'\n');
end
fclose(file);

%generate index files to use on extracting inputs
mkdir('../results/index');
file1 = fopen('../results/index/gesture_index.txt','w');
for i = 1:ends-start+1
    fprintf(file1,strcat('../results/gesture/crop/', num2str(i),'.png+',num2str(i),'\n'));
end
fclose(file1);

end

%generate index file for gesture network output
file2 = fopen('../results/index/bg_op_index.txt','w');
path = pwd;
home = strsplit(path,'core_files');
home = home{1,1};
for i = 1:ends-start+1
    for j = 1:100
        fprintf(file2,strcat(home, 'results/gesture/result_bg_op/', num2str(i),'.h5','\n'));
     end
end
fclose(file2);





    
