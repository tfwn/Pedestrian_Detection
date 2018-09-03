#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // 定义一些数据
#include "my_svm.h" // MySVM继承自CvSVM的类

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
				//第一次归一化求的是平方和
				for (i = hog_index; i<(hog_index + (N_BINS * 4)); i++)
					sum += hog[i] * hog[i];
				//第2次归一化是在第1次的基础上继续求平和和
				scale = 1.f / (sqrt(sum) + sz*0.1f);
				thresh = L2HysThreshold;
				for (i = hog_index, sum = 0; i<(hog_index + (N_BINS * 4)); i++)
				{
					hog[i] = min(hog[i] * scale, thresh);
					sum += hog[i] * hog[i];
				}
				//最终归一化结果
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

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

			  //若TRAIN为true，重新训练分类器
	if (TRAIN)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
										//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
		char saveName[256];
						   //依次读取正样本图片，生成HOG描述子
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
													 //  resize(src,src,Size(64,128));
				}
				sprintf(saveName, "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos64x128\\pos%06d.jpg", num);//生成裁剪出的负样本图片的文件名
				imwrite(saveName, src);//保存文件
			}
			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
												   //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));
									  // 训练 32x64 图片
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表
														 //依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);//读取图片
										  //resize(src,src,Size(64,128));
										  // 训练 32x64 图片
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				vector<float> descriptors;//HOG描述子向量
				hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

														  //cout<<"描述子维数："<<descriptors.size()<<endl;

														  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}



		//输出样本的HOG特征向量矩阵到文件
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		//CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型
	}

	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}


#elif defined train_image_w64x128_b16x16_s16x16_cv

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(16, 16), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

			  //若TRAIN为true，重新训练分类器
	if (TRAIN)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
										//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

						   //依次读取正样本图片，生成HOG描述子
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
													 //  resize(src,src,Size(64,128));
				}
			}
			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
												   //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));
									  // 训练 32x64 图片
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表
														 //依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);//读取图片
				vector<float> descriptors;//HOG描述子向量
				hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型
	}

	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	ofstream fout("D:\\work\\mygit\\Pedestrian_Detection\\HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}

#elif defined train_image_w64x128_b16x16_s16x16_mv

	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
			  //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	int win_w = 64, win_h = 128;
	int block_size = 16;
	int stride_size = 16;
	int cell_size = 8;
	//HOG检测器，用来计算HOG描述子的
	HOGDescriptor hog(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);
	int hog_size = ((win_w - block_size + stride_size) / stride_size)*((win_h - block_size + stride_size) / stride_size) * 4 * 9;

	//若TRAIN为true，重新训练分类器
	if (TRAIN)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
										//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

		float* hog_des = new float[hog_size];

		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
													 //  resize(src,src,Size(64,128));
				}
			}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = hog_size;// descriptors.size();//HOG描述子的维数
										 //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = hog_des[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));
									  // 训练 32x64 图片
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = hog_des[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表
														 //依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);//读取图片
										  //resize(src,src,Size(64,128));
										  // 训练 32x64 图片
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				unsigned char* pImg_data = src.data;
				rectangle_t roi = { 0, 0, win_w, win_h };
				openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);
				//cout<<"描述子维数："<<descriptors.size()<<endl;

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = hog_des[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}





		//输出样本的HOG特征向量矩阵到文件
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型
	}

	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量，将该向量前面乘以 - 1。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中，
	//注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，
	//在HOG的检测函数中，使用rho+alpha*sv*another如果为正的话是正样本，所以需要将后者变为负数之后保存起来
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子

	HOGDescriptor myHOG(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);//HOG检测器，用来计算HOG描述子的
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
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

	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	int win_w = 64, win_h = 128;
	int block_size = 16;
	int stride_size = 8;
	int cell_size = 8;
	//HOG检测器，用来计算HOG描述子的
	HOGDescriptor hog(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);
	int hog_size = ((win_w - block_size + stride_size) / stride_size)*((win_h - block_size + stride_size) / stride_size) * 4 * 9;

	//若TRAIN为true，重新训练分类器
	if (TRAIN)
	{
		string ImgName;//图片名(绝对路径)
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
		//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
		int stride8_hog_size = (block_size / stride_size)*(block_size / stride_size) * (win_w/cell_size) * (win_h/cell_size) * N_BINS;
		float* hog_des = new float[stride8_hog_size];

		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\pos\\" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);//读取图片

			if (CENTRAL_CROP)
			{
				if (src.cols >= 96 && src.rows >= 160)
				{
					src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
													 //  resize(src,src,Size(64,128));
				}
			}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = hog_size;// descriptors.size();//HOG描述子的维数
										 //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = hog_des[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\neg\\" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));
									  // 训练 32x64 图片
									  //if (src.cols > 32 || src.rows > 64)
									  //{
									  //	//resize(src, src, Size(32, 64));
									  //}
			unsigned char* pImg_data = src.data;
			rectangle_t roi = { 0, 0, win_w, win_h };
			openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);

			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = hog_des[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//HardExample负样本的文件名列表
														 //依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num<HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				ImgName = "D:\\work\\mygit\\Pedestrian_Detection\\dataset\\HardExample\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);//读取图片
										  //resize(src,src,Size(64,128));
										  // 训练 32x64 图片
										  //if (src.cols > 32 || src.rows > 64)
										  //{
										  //	resize(src, src, Size(32, 64));
										  //}
				unsigned char* pImg_data = src.data;
				rectangle_t roi = { 0, 0, win_w, win_h };
				openmv_hog_compute(pImg_data, hog_des, &roi, cell_size, stride_size);
				//cout<<"描述子维数："<<descriptors.size()<<endl;

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = hog_des[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}





		//输出样本的HOG特征向量矩阵到文件
		/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

		}
		fout<<endl;
		}*/

		//训练SVM分类器
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		svm.load("D:\\work\\mygit\\Pedestrian_Detection\\SVM_HOG.xml");//从XML文件读取训练好的SVM模型
	}

	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量，将该向量前面乘以 - 1。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中，
	//注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，
	//在HOG的检测函数中，使用rho+alpha*sv*another如果为正的话是正样本，所以需要将后者变为负数之后保存起来
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子

	HOGDescriptor myHOG(Size(win_w, win_h), Size(block_size, block_size), Size(stride_size, stride_size), Size(cell_size, cell_size), 9);//HOG检测器，用来计算HOG描述子的
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
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
	cout << "选择检测方式，如 #define train_image_w64x128_b16x16_s16x16_cv " << endl;
#endif
	/**************读入图片进行HOG行人检测******************/
	Mat src = imread(TestImageFileName);
	vector<Rect> found, found_filtered;//矩形框数组
	cout << "进行多尺度HOG人体检测" << endl;
	myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//对图片进行多尺度行人检测
																			 //src为输入待检测的图片；found为检测到目标区域列表；参数3为程序内部计算为行人目标的阈值，也就是检测到的特征到SVM分类超平面的距离;
																			 //参数4为滑动窗口每次移动的距离。它必须是块移动的整数倍；参数5为图像扩充的大小；参数6为比例系数，即测试图片每次尺寸缩放增加的比例；
																			 //参数7为组阈值，即校正系数，当一个目标被多个窗口检测出来时，该参数此时就起了调节作用，为0时表示不起调节作用。

																			 //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
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
	cout << "找到的矩形框个数：" << found_filtered.size() << endl;

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
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
	waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像


	/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
	////读取测试图片(64*128大小)，并计算其HOG描述子
	//Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
	//将计算好的HOG描述子复制到testFeatureMat矩阵中
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	//用训练好的SVM分类器对测试图片的特征向量进行分类
	//int result = svm.predict(testFeatureMat);//返回类标
	//cout<<"分类结果："<<result<<endl;

	return 0;
}




