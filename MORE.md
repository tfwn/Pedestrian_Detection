# ժҪ��Abstract��
...
# ��Ҫ����

## ���ܣ�Introduction��
...
## �о���״��Related work��
...
## ����������Technique��
 * HOG������ȡ+SVM������ѵ���� �������OpenCVʵ��
 * OpenCV���Ѿ�ʵ����HOG������ȡ��SVM����������㷨������ֱ�ӵ�����غ������ɡ�
 * �������ݿ���� INRIA���ݿ⡣
  > �����ݿ���Ŀǰʹ�����ľ�̬���˼�����ݿ⣬�ṩԭʼͼƬ����Ӧ�ı�ע�ļ���ѵ������������614�ţ�����2416�����ˣ���������1218�ţ����Լ���������288�ţ�����1126�����ˣ���������453�š�ͼƬ������󲿷�Ϊվ�������Ҹ߶ȴ���100�����أ����ֱ�ע���ܲ���ȷ��ͼƬ��Ҫ��Դ��GRAZ-01��������Ƭ��google�����ͼƬ�������Ƚϸߡ���XP����ϵͳ�²���ѵ�����߲���ͼƬ�޷��������������OpenCV������ȡ����ʾ��
 * ������������`INRIAPerson/96X160H96/Train/pos`�ܹ�2416��ͼƬ��ʹ��ʱ���м�ü�����СΪ64x128��ͼƬ��
 * ������������`INRIAPerson/Train/neg`��1218��ͼƬ��ʹ��ʱ��ÿһ��ͼƬ����ü���10�Ŵ�СΪ64x128��ͼƬ��
 * ��������(2416��ͼƬ)�͸�����(12180��ͼƬ)��ȡHOG������Ͷ��SVMѵ������һ��ѵ����ɡ�
 * �õ�һ��ѵ���õ���SVM��������ԭ���ĸ�����(1218��ͼƬ)���м�⣬��������������(��)����ΪHardExample��
 * ��������(2416��ͼƬ)�͸�����(12180��ͼƬ)���ڼ���HardExampleһ����ȡHOG������Ͷ��SVM���еڶ���ѵ����
 * ѵ����ɣ��õ����յ�SVM��������

## ʵ������Experimental Results��
 * ����������������`INRIAPerson/test_64x128_H96/pos`���ܹ�1132��ͼƬ����СΪ70x134����
 * ���Ը�����������`INRIAPerson/test_64x128_H96/neg`��453��ͼƬ����ÿ��ͼƬ����ü���10��ͼƬ����СΪ70x134�����ܹ�4530��ͼƬ��
 * ���Եõ���� TP=795��FN=337��FP=8��TN=4522
  * TP����������������Ϊ���˵�������
  * FN����������������Ϊ�����˵�������
  * FP������������������Ϊ���˵�������
  * TN������������������Ϊ�����˵�������

  <table border="1" style="text-align: center;">
    <tr>
      <td rowspan="2">������</td>
      <td colspan="2">��ʵֵ</td>
    </tr>
      <td>Positive�����ˣ�</td>
      <td>Negative�������ˣ�</td>
    <tr>
      <td>Positive�����ˣ�</td>
      <td>True Positive (TP)</td>
      <td>False Positive (FP)</td>
    </tr>
    <tr>
      <td>Negative�������ˣ�</td>
      <td>False Negative (FN)</td>
      <td>True Negative (TN)</td>
    </tr>
  </table>

# ���ۣ�Conclusion��
...
# �ο����ף�References��
* Dalal, N. and Triggs, B., ��Histograms of oriented gradients for human detection,�� in [Computer Vision and
Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on ], 1, 886�C893 vol. 1 (June).
* �Լ�ѵ��SVM����������HOG���˼�� http://blog.csdn.net/masibuaa/article/details/16105073
* ����Hog������SVM�������������˼�� http://blog.csdn.net/carson2005/article/details/7841443

