#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // ����һЩ����
#include "my_svm.h" // MySVM�̳���CvSVM����

using namespace std;
using namespace cv;

//#define HOG_SIZE ( 1152 )
#define M_PI    3.14159265f
#define N_BINS      (9)
typedef struct _rectangle {
	int16_t x;
	int16_t y;
	int16_t w;
	int16_t h;
} rectangle_t;

short lut64[] = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
8, 9, 10, 11, 12, 13, 14, 14, 15, 16,
17, 18, 18, 19, 20, 21, 22, 22, 23, 24,
25, 25, 26, 27, 27, 28, 29, 30, 30, 31,
32, 32, 33, 33, 34 };
short lut32[] = { 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
45, 46, 47, 47, 48, 49, 50, 50, 51, 52,
53, 53, 54, 54, 55, 56, 56, 57, 57, 58,
58, 59, 59, 59 };
short lut16[] = { 60, 60, 61, 62, 63, 63, 64, 65, 65, 66,
67, 67, 68, 68, 69, 69, 69 };
short lut8[] = { 70, 71, 71, 72, 73, 73, 74, 74 };
short lut4[] = { 75, 75, 76, 76, 77, 78, 78, 79, 79 };
short lut1[] = { 80, 81, 82, 83, 84, 84, 85, 85, 85, 86,
86, 86, 86, 86, 87, 87, 87 };

float _boYiFastAtan2_32f(const int Y, const int X)
{
	int slope = 0;
	int index = 0;
	int i = 0;
	short ang = 0;
	float angle = 0;
	float scale = (float)(M_PI / 180);

	float ax = X * 1000;
	float ay = Y * 1000;
	short dx = 0, dy = 0;
	dx = (short)(ax);
	dy = (short)(ay);

	if (dx == 0)
	{
		// prevent division-by-zero
		if (dy > 0)
			ang = 90;
		else if (dy < 0)
			ang = 90;
		else ang = 0;//ang = 0;
	}
	else
	{
		slope = (64 * dy) / dx;
		if (slope < 0)
		{
			slope = -slope;
		}
		if (slope > 362)
		{
			index = (slope / 64) - 5;
			if (index > 16)
				index = 16; // prevent too large angle
			ang = lut1[index];
		}
		else if (slope > 238)
		{
			index = (slope / 16) - 14;
			ang = lut4[index];
		}
		else if (slope > 175)
		{
			index = (slope / 8) - 22;
			ang = lut8[index];
		}
		else if (slope > 110)
		{
			index = (slope / 4) - 27;
			ang = lut16[index];
		}
		else if (slope > 44)
		{
			index = (slope / 2) - 22;
			ang = lut32[index];
		}
		else
		{
			index = slope;
			ang = lut64[index];
		}
	}
	if (dx < 0)
		ang = 180 - ang;
	if (dy < 0)
		ang = 360 - ang;

	angle = (float)(ang*scale);

	return angle;

}

float* openmv_hog_compute(unsigned char* src, float* hog, rectangle_t *roi, int cell_size, int stride_size)
{
	int x = 0, y = 0, hog_index = 0;
	int i = 0, t = 0, vx = 0, vy = 0;
	int cx = 0, cy = 0;
	float m = 0.0;

	int s = roi->w;
	int w = roi->x + roi->w - 1;
	int h = roi->y + roi->h - 1;

	int block_size = cell_size * 2;
	int x_cells = (roi->w / cell_size);
	int y_cells = (roi->h / cell_size);
	int hog_size = ((roi->w - block_size + stride_size) / stride_size)*((roi->h - block_size + stride_size) / stride_size) * 4 * 9;

	// TODO: Assert row->w/h >= cell_size *2;
	//memset(hog, 0, hog_size * sizeof(float));
	memset(hog, 0, (block_size/ stride_size)*(block_size / stride_size) * x_cells * y_cells * N_BINS * sizeof(float));

	//2. Finding Image Gradients
	for (y = roi->y, hog_index = 0; y<h; y += stride_size)
	{
		for (x = roi->x; x<w; x += stride_size)
		{
			float k = 0.0f;
			for (cy = 0; cy<block_size; cy++)
			{
				for (cx = 0; cx<block_size; cx++)
				{
					if ((y + cy) > 0 && (y + cy) < h && (x + cx) > 0 && (x + cx) < w)
					{
						// Find horizontal/vertical direction
						vx = src[(y + cy + 0)*s + (x + cx + 1)] - src[(y + cy - 0)*s + (x + cx - 1)];
						vy = src[(y + cy + 1)*s + (x + cx + 0)] - src[(y + cy - 1)*s + (x + cx - 0)];

						// Find magnitude
						//float m = fast_sqrtf(vx*vx + vy*vy);
						m = sqrtf(vx * vx + vy * vy);
						//m = abs(vx) + abs(vy); // from boyi
						if (((int)m) > 1)
						{
							k += m * m;
							// Find and quantize gradient degree
							// TODO atan2f is swapped for visualization
							//int t = ((int)fast_fabsf((atan2f(vx, vy)*180.0f / M_PI))) / 20;
							t = ((int)fabsf((atan2f(vx, vy) * 180.0f / M_PI))) / 20;
							//t = (int)(_boYiFastAtan2_32f(vx, vy) * 9.0f / M_PI);
							t = (t == 9) ? 0 : t;

							//hog[hog_index+((cy/cell_size) * x_cells + (cx/cell_size)) * N_BINS + t] += m;
							hog[hog_index + (((cy / cell_size) * 2 + (cx / cell_size)) * N_BINS) + t] += m;
						}
					}
				}
#if 0
				float sum = 0, scale, thresh;
				float L2HysThreshold = 0.2;
				int sz = 36;
				//��һ�ι�һ�������ƽ����
				for (i = hog_index; i<(hog_index + (N_BINS * 4)); i++)
					sum += hog[i] * hog[i];
				//��2�ι�һ�����ڵ�1�εĻ����ϼ�����ƽ�ͺ�
				scale = 1.f / (sqrt(sum) + sz*0.1f);
				thresh = L2HysThreshold;
				for (i = hog_index, sum = 0; i<(hog_index + (N_BINS * 4)); i++)
				{
					hog[i] = min(hog[i] * scale, thresh);
					sum += hog[i] * hog[i];
				}
				//���չ�һ�����
				scale = 1.f / (sqrt(sum) + 1e-3f);
				for (i = hog_index; i<(hog_index + (N_BINS * 4)); i++)
					hog[i] *= scale;
#endif
			}

			// Normalize the last block
			k = sqrtf(k);
			for (i = hog_index; i<(hog_index + (N_BINS * 4)); i++)
			{
				hog[i] = hog[i] / k;
			}

			hog_index += (N_BINS * 4);
		}
	}

	return hog;
}








int main(int argc, char const *argv[])
{

#ifdef train_image_w64x128_b16x16_s8x8_cv	

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

			  //��TRAINΪtrue������ѵ��������
	if (TRAIN)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(PosSamListFile);//������ͼƬ���ļ����б�
										//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg(NegSamListFile);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����
		char saveName[256];
						   //���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
													 //  resize(src,src,Size(64,128));
				}
				sprintf(saveName, "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos64x128\\pos%06d.jpg", num);//���ɲü����ĸ�����ͼƬ���ļ���
				imwrite(saveName, src);//�����ļ�
			}
			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
												   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));
									  // ѵ�� 32x64 ͼƬ
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������

		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample���������ļ����б�
														 //���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//����HardExample��������·����
				Mat src = imread(ImgName);//��ȡͼƬ
										  //resize(src,src,Size(64,128));
										  // ѵ�� 32x64 ͼƬ
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				vector<float> descriptors;//HOG����������
				hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

														  //cout<<"������ά����"<<descriptors.size()<<endl;

														  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}



		//���������HOG�������������ļ�
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		//CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


#elif defined train_image_w64x128_b16x16_s16x16_cv

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(16, 16), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

			  //��TRAINΪtrue������ѵ��������
	if (TRAIN)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(PosSamListFile);//������ͼƬ���ļ����б�
										//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg(NegSamListFile);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

						   //���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
													 //  resize(src,src,Size(64,128));
				}
			}
			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
												   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));
									  // ѵ�� 32x64 ͼƬ
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������

		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample���������ļ����б�
														 //���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//����HardExample��������·����
				Mat src = imread(ImgName);//��ȡͼƬ
				vector<float> descriptors;//HOG����������
				hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}

#elif defined train_image_w64x128_b16x16_s16x16_mv

	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
			  //��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	int win_w = 64, win_h = 128;
	int block_size = 16;
	int stride_size = 16;
	int cell_size = 8;
	//HOG���������������HOG�����ӵ�
	HOGDescriptor hog(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);
	int hog_size = ((win_w - block_size + stride_size) / stride_size)*((win_h - block_size + stride_size) / stride_size) * 4 * 9;

	//��TRAINΪtrue������ѵ��������
	if (TRAIN)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(PosSamListFile);//������ͼƬ���ļ����б�
										//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg(NegSamListFile);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

		float* hog_des = new float[hog_size];

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
													 //  resize(src,src,Size(64,128));
				}
			}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"������ά����"<<descriptors.size()<<endl;

			//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = hog_size;// descriptors.size();//HOG�����ӵ�ά��
										 //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = hog_des[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));
									  // ѵ�� 32x64 ͼƬ
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"������ά����"<<descriptors.size()<<endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = hog_des[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������

		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample���������ļ����б�
														 //���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//����HardExample��������·����
				Mat src = imread(ImgName);//��ȡͼƬ
										  //resize(src,src,Size(64,128));
										  // ѵ�� 32x64 ͼƬ
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				unsigned char* pImg_data = src.data;
				rectangle_t roi = { 0, 0, win_w, win_h };
				openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);
				//cout<<"������ά����"<<descriptors.size()<<endl;

				//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = hog_des[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}





		//���������HOG�������������ļ�
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ������������������ǰ����� - 1��֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat�У�
	//ע����Ϊsvm.predictʹ�õ���alpha*sv*another-rho�����Ϊ���Ļ�����Ϊ����������
	//��HOG�ļ�⺯���У�ʹ��rho+alpha*sv*another���Ϊ���Ļ�����������������Ҫ�����߱�Ϊ����֮�󱣴�����
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����

	HOGDescriptor myHOG(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);//HOG���������������HOG�����ӵ�
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		if (i % 4 == 0)
		{
			fout << "\n" << endl;
		}
		fout << myDetector[i] << endl;
		fout << "f," << endl;
	}
#elif defined train_image_w64x128_b16x16_s8x8_mv

	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	int win_w = 64, win_h = 128;
	int block_size = 16;
	int stride_size = 8;
	int cell_size = 8;
	//HOG���������������HOG�����ӵ�
	HOGDescriptor hog(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);
	int hog_size = ((win_w - block_size + stride_size) / stride_size)*((win_h - block_size + stride_size) / stride_size) * 4 * 9;

	//��TRAINΪtrue������ѵ��������
	if (TRAIN)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos(PosSamListFile);//������ͼƬ���ļ����б�
		//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg(NegSamListFile);//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����
		int stride8_hog_size = (block_size / stride_size)*(block_size / stride_size) * (win_w/cell_size) * (win_h/cell_size) * N_BINS;
		float* hog_des = new float[stride8_hog_size];

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
													 //  resize(src,src,Size(64,128));
				}
			}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"������ά����"<<descriptors.size()<<endl;

			//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = hog_size;// descriptors.size();//HOG�����ӵ�ά��
										 //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = hog_des[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));
									  // ѵ�� 32x64 ͼƬ
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"������ά����"<<descriptors.size()<<endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = hog_des[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������

		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample���������ļ����б�
														 //���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//����HardExample��������·����
				Mat src = imread(ImgName);//��ȡͼƬ
										  //resize(src,src,Size(64,128));
										  // ѵ�� 32x64 ͼƬ
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				unsigned char* pImg_data = src.data;
				rectangle_t roi = { 0, 0, win_w, win_h };
				openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);
				//cout<<"������ά����"<<descriptors.size()<<endl;

				//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = hog_des[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}





		//���������HOG�������������ļ�
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ������������������ǰ����� - 1��֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat�У�
	//ע����Ϊsvm.predictʹ�õ���alpha*sv*another-rho�����Ϊ���Ļ�����Ϊ����������
	//��HOG�ļ�⺯���У�ʹ��rho+alpha*sv*another���Ϊ���Ļ�����������������Ҫ�����߱�Ϊ����֮�󱣴�����
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����

	HOGDescriptor myHOG(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);//HOG���������������HOG�����ӵ�
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		if (i % 4 == 0)
		{
			fout << "\n" << endl;
		}
		fout << myDetector[i] << endl;
		fout << "f," << endl;
	}

#else
	cout << "ѡ���ⷽʽ���� #define train_image_w64x128_b16x16_s16x16_cv " << endl;
#endif
	/**************����ͼƬ����HOG���˼��******************/
	Mat src = imread(TestImageFileName);
	vector<Rect> found, found_filtered;//���ο�����
	cout << "���ж�߶�HOG������" << endl;
	myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//��ͼƬ���ж�߶����˼��
																			 //srcΪ���������ͼƬ��foundΪ��⵽Ŀ�������б�����3Ϊ�����ڲ�����Ϊ����Ŀ�����ֵ��Ҳ���Ǽ�⵽��������SVM���೬ƽ��ľ���;
																			 //����4Ϊ��������ÿ���ƶ��ľ��롣�������ǿ��ƶ���������������5Ϊͼ������Ĵ�С������6Ϊ����ϵ����������ͼƬÿ�γߴ��������ӵı�����
																			 //����7Ϊ����ֵ����У��ϵ������һ��Ŀ�걻������ڼ�����ʱ���ò�����ʱ�����˵������ã�Ϊ0ʱ��ʾ����������á�

																			 //�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
		printf("x=%d y=%d w=%d h=%d\n", found[i].x, found[i].y, found[i].width, found[i].height);

	}
	cout << "�ҵ��ľ��ο������" << found_filtered.size() << endl;

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}

	imwrite("D:\\work\\mygit\\Pedestrian_Detection\\ImgProcessed.jpg", src);
	namedWindow("src", 0);
	imshow("src", src);
	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��


	/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
	////��ȡ����ͼƬ(64*128��С)����������HOG������
	//Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������
	//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
	//int result = svm.predict(testFeatureMat);//�������
	//cout<<"��������"<<result<<endl;

	return 0;
}




