#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include "my_svm.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
  VideoCapture capture;
  if( argc == 1 )
  {
    capture.open("video.avi");
    if(!capture.isOpened()){
      printf("Usage: %s (<image_filename> | <video_filename>)\n",argv[0]);
      return 0;
    }
  } else {
    capture.open(argv[1]);
    if(!capture.isOpened()){
      printf("Usage: %s <video_filename>\n",argv[0]);
      return 0;
    }
  }

  //��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
  //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG�����ӵ�
  int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
  MySVM svm;//SVM������
  svm.load("SVM_HOG.xml");

  /*************************************************************************************************
  ����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
  ��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
  ��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
  �Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
  ***************************************************************************************************/
  DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
  int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
  cout<<"֧������������"<<supportVectorNum<<endl;

  Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
  Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
  Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

  //��֧�����������ݸ��Ƶ�supportVectorMat������
  for(int i=0; i<supportVectorNum; i++)
  {
    const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
    for(int j=0; j<DescriptorDim; j++)
    {
      //cout<<pData[j]<<" ";
      supportVectorMat.at<float>(i,j) = pSVData[j];
    }
  }

  //��alpha���������ݸ��Ƶ�alphaMat��
  double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
  for(int i=0; i<supportVectorNum; i++)
  {
    alphaMat.at<float>(0,i) = pAlphaData[i];
  }

  //����-(alphaMat * supportVectorMat),����ŵ�resultMat��
  //gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
  resultMat = -1 * alphaMat * supportVectorMat;

  //�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
  vector<float> myDetector;
  //��resultMat�е����ݸ��Ƶ�����myDetector��
  for(int i=0; i<DescriptorDim; i++)
  {
    myDetector.push_back(resultMat.at<float>(0,i));
  }
  //������ƫ����rho���õ������
  myDetector.push_back(svm.get_rho());
  cout<<"�����ά����"<<myDetector.size()<<endl;
  //����HOGDescriptor�ļ����
  HOGDescriptor myHOG;
  myHOG.setSVMDetector(myDetector);
  //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  //	//�������Ӳ������ļ�
  //	ofstream fout("HOGDetectorForOpenCV.txt");
  //	for(int i=0; i<myDetector.size(); i++)
  //	{
  //		fout<<myDetector[i]<<endl;
  //	}

  //VideoCapture capture(argv[1]);
  //if(!capture.isOpened())
  //  return 1;
  double rate=capture.get(CV_CAP_PROP_FPS);
  bool stop(false);
  Mat frame;

  namedWindow("Video");
  int delay = 1000/rate;

  while(!stop)
  {
    if(!capture.read(frame))
      break;
    Mat src=frame;

    vector<Rect> found, found_filtered;//���ο�����
    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//��ͼƬ���ж�߶����˼��

    //�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
    for(int i=0; i < found.size(); i++)
    {
        Rect r = found[i];
        int j=0;
        for(; j < found.size(); j++)
            if(j != i && (r & found[j]) == r)
                break;
        if( j == found.size())
            found_filtered.push_back(r);
    }

    //�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
    for(int i=0; i<found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
    }

    imshow("Video",src);

    if(waitKey(delay)>=0)
      stop=true;
  }
  capture.release();
}

