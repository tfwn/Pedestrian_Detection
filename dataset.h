


#ifndef DATASET_H
#define DATASET_H

#define PosSamNO 2416  //正样本个数
#define NegSamNO 12180    //负样本个数

#define PosSamListFile "D:\\work\\mygit\\Pedestrian_Detection\\INRIAPerson96X160PosList.txt" //正样本图片的文件名列表
#define NegSamListFile "D:\\work\\mygit\\Pedestrian_Detection\\NoPersonFromINRIAList.txt" //负样本图片的文件名列表

#define TRAIN true//false //true   //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体


#define train_image_w64x128_b16x16_s8x8_cv
//#define train_image_w64x128_b16x16_s16x16_cv
//#define train_image_w64x128_b16x16_s8x8_mv
//#define train_image_w64x128_b16x16_s16x16_mv

#define HardExampleListFile "D:\\work\\mygit\\Pedestrian_Detection\\HardExample_FromINRIA_NegList.txt"
//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 0

#define TermCriteriaCount 5000  //迭代终止条件，当迭代满50000次或误差小于FLT_EPSILON时停止迭代

#define TestImageFileName "D:\\work\\mygit\\Pedestrian_Detection\\000001.jpg"  //训练完成后读入一张图片来测试效果


#endif


