# HOG+SVM进行行人检测
> 运行环境为：Ubuntu 16.04 && OpenCV 2.4.13
> 代码参考自：http://blog.csdn.net/masibuaa/article/details/16105073
> INRIA Person数据库下载：
【[相关说明](http://pascal.inrialpes.fr/data/human/)】
>【直接下载地址（970M）】ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar

## 开始前的准备工作
开始前建立文件夹用于存储正负样本和HardExample，正样本图片直接复制INRIA中的正样本图片，负样本图片通过裁剪得到。
```shell
$ mkdir -p dataset/pos dataset/neg dataset/HardExample
$ cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
```
## 编译出可执行文件
```shell
$ cmake .
$ make
```
> 也可以在命令行使用`g++`编译出可执行文件，例如
```shell
$ g++ -o CropImage crop_image.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o SvmTrainUseHog main.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o GetHardExample find_save_HardExample.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o ImageDetect image_detect.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o VideoDetect video_detect.cpp $(pkg-config opencv --cflags --libs)
$ g++ -o PeopleDetect peopledetect.cpp $(pkg-config opencv --cflags --libs)
```

## 第一步，裁剪出负样本图片
INRIA中有1218张负样本图片，`CropImage`从每一张图片中随机裁剪出10张大小为64x128的图片，最终总共会得到12180张图片，存储在dataset/neg文件夹中。上面已经编译出了可执行文件，直接通过`CropImage`裁剪出负样本图片。
```shell
$ ./CropImage
```
## 第二步，使用正负样本进行训练
先修改 `dataset.h` 里面参数，将 `TRAIN` 由 `false` 改为 `true` , 以进行训练，参数修改后需通过 `make` 重新编译可执行文件，然后通过 `SvmTrainUseHog` 开始训练。
```shell
$ make
$ ./SvmTrainUseHog
```
 到这里已经得到了 `SVM_HOG.xml` 可以用来进行检测，但是检测效果不太好，所以下面加入 HardExample 来进行进一步训练。

## 第三步，得到HardExample
通过 `GetHardExample` 从 INRIA 原始的负样本图片中检测出 HardExample ，图片会保存到 dataset/HardExample
```shell
$ ./GetHardExample
```
## 第四步，将正负样本和HardExample一起重新进行训练
将 HardExample 图片列表写入文件 `HardExample_FromINRIA_NegList.txt` ，
修改 `dataset.h` 里面的参数，将 `HardExampleNO` 由 `0` 改为第三步中得到的图片数目，修改后通过 `make` 重新编译可执行文件，最后通过 `SvmTrainUseHog` 重新训练。
```shell
$ ls dataset/HardExample/ >HardExample_FromINRIA_NegList.txt
$ make
$ ./SvmTrainUseHog
```
## 至此训练完成。
检测示例图片：
![检测示例图片](https://github.com/icsfy/Pedestrian_Detection/raw/master/ImgProcessed.jpg)
## 其它说明
* `SVM_HOG.xml`为最终训练好的SVM分类器
* `ImageDetect`可对图片进行检测
* `VideoDetect`可对视频进行检测
* `PeopleDetect`为OpenCV默认参数的行人检测程序

## [More](https://github.com/icsfy/Pedestrian_Detection/blob/master/MORE.md)















****************************************************************************************
源文件：
	https://github.com/icsfy/Pedestrian_Detection

参考：https://blog.csdn.net/gojawee/article/category/6778915/1

1.  将 INRIAPerson.tar  下载解压。 （http://pascal.inrialpes.fr/data/human/） 
2.  	mkdir -p dataset/pos dataset/neg dataset/HardExample
	cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
	Windows下手动生成几个文件夹并将图片放到pos文件夹下
	
3. 使用 vs2015 + opencv2.4.13 编译相关文件
    1）编译 crop_image.cpp + dataset.h + my_svm.h，生成用于裁剪负样本图片的程序。
    2）编译 main.cpp + dataset.h + my_svm.h，生成用于训练的程序。
    3）编译 find_save_HardExample.cpp + dataset.h + my_svm.h，生成用于获取HardExample的程序。
    4）编译 image_detect.cpp + dataset.h + my_svm.h，用于生成测试的程序。
    5）peopledetect.cpp ，生成的opencv自带的行人检测程序。
    6）video_detect.cpp，生成用于检测视频中行人的检测程序。


可能出现的问题：
1.
修改训练窗口大小不是 64x128, 可能会出现
OpenCV Error: Assertion failed <checkDetectorSize<>> in cv::HOGDescriptor::setSVMDetector,  file ..\..\..\..\opencv\modules\objdetect\src\hog.cpp, line89
解决：在mian.cpp函数中增加hog检测子的大小定义。
#ifdef train_image_64x128		
	HOGDescriptor myHOG;
#else
	HOGDescriptor myHOG(Size(32, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
#endif

2. 训练样本或者hard样本时
OpenCV Error: Assertion failed (ssize.area() > 0) in cv::resize, file C:\build\2_4_winpack-build-win64-vc14\opencv\modules\imgproc\src\imgwarp.cpp, line 1968
分析：
1.空路径空文件等
2.超大文件，数据溢出
3.检测出的窗口大小为 0的情况下去resize

3.
OpenCV Error: Assertion failed (dims <= 2 && step[0] > 0) in cv::Mat::locateROI, file C:
分析：
出错的地方位于opencv内部的“断言”错误。看相关代码，是在调用opencv相关函数的时候参数不符合规则。
也就是说，代码在访问某个Mat矩阵时越界了，所以报错。
解决：查看 D:\\work\\git\\Pedestrian_Detection\\HardExample_FromINRIA_NegList.txt 等相关文件中的编码方式是否是 UTF-8





















