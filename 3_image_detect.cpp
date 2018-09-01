#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "my_svm.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  Mat img;
  FILE* f = 0;
  char _filename[1024];

  if( argc == 1 )
  {
      printf("Usage: %s (<image_filename> | <image_list>.txt)\n",argv[0]);
      return 0;
  }

  img = imread(argv[1]);

  if( img.data )
  {
      strcpy(_filename, argv[1]);
  }
  else
  {
      f = fopen(argv[1], "rt");
      if(!f)
      {
          fprintf( stderr, "ERROR: the specified file could not be loaded\n");
          return -1;
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

  namedWindow("people detector", 1);

  for(;;)
  {
      char* filename = _filename;
      if(f)
      {
          if(!fgets(filename, (int)sizeof(_filename)-2, f))
              break;
          //while(*filename && isspace(*filename))
          //  ++filename;
          if(filename[0] == '#')
              continue;
          int l = (int)strlen(filename);
          while(l > 0 && isspace(filename[l-1]))
              --l;
          filename[l] = '\0';
          img = imread(filename);
      }
      printf("%s:\n", filename);
      if(!img.data)
          continue;

      fflush(stdout);
      vector<Rect> found, found_filtered;
      double t = (double)getTickCount();
      // run the detector with default parameters. to get a higher hit-rate
      // (and more false alarms, respectively), decrease the hitThreshold and
      // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
      myHOG.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
      t = (double)getTickCount() - t;
      printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
      size_t i, j;
      for( i = 0; i < found.size(); i++ )
      {
          Rect r = found[i];
          for( j = 0; j < found.size(); j++ )
              if( j != i && (r & found[j]) == r)
                  break;
          if( j == found.size() )
              found_filtered.push_back(r);
      }
      for( i = 0; i < found_filtered.size(); i++ )
      {
          Rect r = found_filtered[i];
          // the HOG detector returns slightly larger rectangles than the real objects.
          // so we slightly shrink the rectangles to get a nicer output.
          r.x += cvRound(r.width*0.1);
          r.width = cvRound(r.width*0.8);
          r.y += cvRound(r.height*0.07);
          r.height = cvRound(r.height*0.8);
          rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
      }
      imshow("people detector", img);
      int c = waitKey(0) & 255;
      if( c == 'q' || c == 'Q' || !f)
          break;
  }
  if(f)
      fclose(f);
  return 0;
}

