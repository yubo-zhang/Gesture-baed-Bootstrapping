

#test results
layer = 'prob2'
image = 'data'
gt = 'label'
model = '../caffe/models/apperance/VGG_VOC2012ext_iter_'+sys.argv[1]+'.caffemodel'
test_net = caffe.Net('../caffe/models/appearance/VGG_VOC2012test.prototxt',model,caffe.TEST)
cur_path = os.getcwd()
a = cur_path.split('core_files')
result_dir = os.path.join(a[0] + 'results/appearance/images')
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
txt_dir = os.path.join(a[0] + 'results/appearance/F1score.txt')
test_num = len([name for name in os.listdir('../test_images/RGBimages') if os.path.isfile(os.path.join('../test_images/RGBimages',name))]) #number of images to test
eul_test.F1_measure(test_net,result_dir,txt_dir,test_num,layer,image ,gt)
