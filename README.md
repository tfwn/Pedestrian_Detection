# HOG+SVM�������˼��
> ���л���Ϊ��Ubuntu 16.04 && OpenCV 2.4.13
> ����ο��ԣ�http://blog.csdn.net/masibuaa/article/details/16105073
> INRIA Person���ݿ����أ�
��[���˵��](http://pascal.inrialpes.fr/data/human/)��
>��ֱ�����ص�ַ��970M����ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar

## ��ʼǰ��׼������
��ʼǰ�����ļ������ڴ洢����������HardExample��������ͼƬֱ�Ӹ���INRIA�е�������ͼƬ��������ͼƬͨ���ü��õ���
```shell
$ mkdir -p dataset/pos dataset/neg dataset/HardExample
$ cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
```
## �������ִ���ļ�
```shell
$ cmake .
$ make
```
> Ҳ������������ʹ��`g++`�������ִ���ļ�������
```shell
$ g++ -o CropImage crop_image.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o SvmTrainUseHog main.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o GetHardExample find_save_HardExample.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o ImageDetect image_detect.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o VideoDetect video_detect.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o PeopleDetect peopledetect.cpp $(pkg-config opencv --cflags --libs)
```

## ��һ�����ü���������ͼƬ
INRIA����1218�Ÿ�����ͼƬ��`CropImage`��ÿһ��ͼƬ������ü���10�Ŵ�СΪ64x128��ͼƬ�������ܹ���õ�12180��ͼƬ���洢��dataset/neg�ļ����С������Ѿ�������˿�ִ���ļ���ֱ��ͨ��`CropImage`�ü���������ͼƬ��
```shell
$ ./CropImage
```
## �ڶ�����ʹ��������������ѵ��
���޸� `dataset.h` ����������� `TRAIN` �� `false` ��Ϊ `true` , �Խ���ѵ���������޸ĺ���ͨ�� `make` ���±����ִ���ļ���Ȼ��ͨ�� `SvmTrainUseHog` ��ʼѵ����
```shell
$ make
$ ./SvmTrainUseHog
```
 �������Ѿ��õ��� `SVM_HOG.xml` �����������м�⣬���Ǽ��Ч����̫�ã������������ HardExample �����н�һ��ѵ����

## ���������õ�HardExample
ͨ�� `GetHardExample` �� INRIA ԭʼ�ĸ�����ͼƬ�м��� HardExample ��ͼƬ�ᱣ�浽 dataset/HardExample
```shell
$ ./GetHardExample
```
## ���Ĳ���������������HardExampleһ�����½���ѵ��
�� HardExample ͼƬ�б�д���ļ� `HardExample_FromINRIA_NegList.txt` ��
�޸� `dataset.h` ����Ĳ������� `HardExampleNO` �� `0` ��Ϊ�������еõ���ͼƬ��Ŀ���޸ĺ�ͨ�� `make` ���±����ִ���ļ������ͨ�� `SvmTrainUseHog` ����ѵ����
```shell
$ ls dataset/HardExample/ >HardExample_FromINRIA_NegList.txt
$ make
$ ./SvmTrainUseHog
```
## ����ѵ����ɡ�
���ʾ��ͼƬ��
![���ʾ��ͼƬ](https://github.com/icsfy/Pedestrian_Detection/raw/master/ImgProcessed.jpg)
## ����˵��
* `SVM_HOG.xml`Ϊ����ѵ���õ�SVM������
* `ImageDetect`�ɶ�ͼƬ���м��
* `VideoDetect`�ɶ���Ƶ���м��
* `PeopleDetect`ΪOpenCVĬ�ϲ��������˼�����

## [More](https://github.com/icsfy/Pedestrian_Detection/blob/master/MORE.md)















****************************************************************************************
Դ�ļ���
	https://github.com/icsfy/Pedestrian_Detection

�ο���https://blog.csdn.net/gojawee/article/category/6778915/1

1.  �� INRIAPerson.tar  ���ؽ�ѹ�� ��http://pascal.inrialpes.fr/data/human/�� 
2.  	mkdir -p dataset/pos dataset/neg dataset/HardExample
	cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
	Windows���ֶ����ɼ����ļ��в���ͼƬ�ŵ�pos�ļ�����
	
3. ʹ�� vs2015 + opencv2.4.13 ��������ļ�
    1������ crop_image.cpp + dataset.h + my_svm.h���������ڲü�������ͼƬ�ĳ���
    2������ main.cpp + dataset.h + my_svm.h����������ѵ���ĳ���
    3������ find_save_HardExample.cpp + dataset.h + my_svm.h���������ڻ�ȡHardExample�ĳ���
    4������ image_detect.cpp + dataset.h + my_svm.h���������ɲ��Եĳ���
    5��peopledetect.cpp �����ɵ�opencv�Դ������˼�����
    6��video_detect.cpp���������ڼ����Ƶ�����˵ļ�����


���ܳ��ֵ����⣺
1.
�޸�ѵ�����ڴ�С���� 64x128, ���ܻ����
OpenCV Error: Assertion failed <checkDetectorSize<>> in cv::HOGDescriptor::setSVMDetector,  file ..\..\..\..\opencv\modules\objdetect\src\hog.cpp, line89
�������mian.cpp����������hog����ӵĴ�С���塣
#ifdef train_image_64x128		
	HOGDescriptor myHOG;
#else
	HOGDescriptor myHOG(Size(32, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
#endif

2. ѵ����������hard����ʱ
OpenCV Error: Assertion failed (ssize.area() > 0) in cv::resize, file C:\build\2_4_winpack-build-win64-vc14\opencv\modules\imgproc\src\imgwarp.cpp, line 1968
������
1.��·�����ļ���
2.�����ļ����������
3.�����Ĵ��ڴ�СΪ 0�������ȥresize

3.
OpenCV Error: Assertion failed (dims <= 2 && step[0] > 0) in cv::Mat::locateROI, file C:
������
����ĵط�λ��opencv�ڲ��ġ����ԡ����󡣿���ش��룬���ڵ���opencv��غ�����ʱ����������Ϲ���
Ҳ����˵�������ڷ���ĳ��Mat����ʱԽ���ˣ����Ա���
������鿴 D:\\work\\git\\Pedestrian_Detection\\HardExample_FromINRIA_NegList.txt ������ļ��еı��뷽ʽ�Ƿ��� UTF-8





















