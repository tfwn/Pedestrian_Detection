


#ifndef DATASET_H
#define DATASET_H

#define PosSamNO 2416  //����������
#define NegSamNO 12180    //����������

#define PosSamListFile "D:\\work\\git\\Pedestrian_Detection\\INRIAPerson96X160PosList.txt" //������ͼƬ���ļ����б�
#define NegSamListFile "D:\\work\\git\\Pedestrian_Detection\\NoPersonFromINRIAList.txt" //������ͼƬ���ļ����б�

#define TRAIN true//false //true   //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define CENTRAL_CROP true   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����


//#define train_image_w64x128_b16x16_s8x8_cv
//#define train_image_w64x128_b16x16_s16x16_cv
#define train_image_w64x128_b16x16_s16x16_mv

#define HardExampleListFile "D:\\work\\git\\Pedestrian_Detection\\HardExample_FromINRIA_NegList.txt"
//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ
#define HardExampleNO 0

#define TermCriteriaCount 50000  //������ֹ��������������50000�λ����С��FLT_EPSILONʱֹͣ����

#define TestImageFileName "D:\\work\\git\\Pedestrian_Detection\\Test.jpg"  //ѵ����ɺ����һ��ͼƬ������Ч��


#endif


