#include "detector.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void Detector::Preprocessing(Mat &img)
{
    float mean[] = {0.40559885502486, -0.019621851500929, 0.026953143125972};
    float std[] = {0.26126178026709, 0.049694558439293, 0.071862255292542};
    img.convertTo(img, CV_32F, 1.0f/255.0f);
    cvtColor(img, img, COLOR_BGR2YCrCb);

    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            Vec3f pixel = img.at<Vec3f>(y, x);
            for (int z = 0; z < 3; z++)
            {
                pixel[z] = (pixel[z] - mean[z]) / std[z];
            }
        }
    }
}

Detector::Detector(std::shared_ptr<Classifier> classifier_,
                   cv::Size window_size_, int dx_ = 1, int dy_ = 1,
                   double scale_ = 1.2, int min_neighbours_ = 3,
                   bool group_rect_ = false)
    : classifier(classifier_),
      window_size(window_size_),
      dx(dx_),
      dy(dy_),
      scale(scale_),
      min_neighbours(min_neighbours_),
      group_rect(group_rect_)
{}

void Detector::CreateImagePyramid(const cv::Mat &img, std::vector<Mat> &pyramid,
                                  std::vector<float> &scales)
{
    pyramid.clear();
    pyramid.push_back(img);
    Mat resizedImg;
    float scaleFactor = 1.0f;
    while (img.cols > window_size.width && img.rows > window_size.height)
    {
        resize(img, resizedImg,
               Size((int)(img.cols / scale), (int)(img.rows / scale)),
               0, 0, INTER_LINEAR);
        pyramid.push_back(resizedImg.clone());
        scales.push_back(scaleFactor);        
        scaleFactor *= scale;
    }    
}

void Detector::Detect(Mat &layer, vector<int> &labels,
        vector<double> &scores, vector<Rect> &rects,
        const float scaleFactor,
        const float detectorThreshold, 
        const double mergeRectThreshold)
{
  vector<Rect> layerRect;
  for (int y = 0; y < layer.rows - window_size.height + 1; y += dy)
  {
    for (int x = 0; x < layer.cols - window_size.width + 1; x += dx)
    {
      Rect rect(x, y, window_size.width, window_size.height);
      Mat window = layer(rect);

      Classifier::Result result = classifier->Classify(window);
      if (fabs(result.confidence) < detectorThreshold && result.label == 1)
      {
        labels.push_back(result.label);
        scores.push_back(result.confidence);
        layerRect.push_back(
          Rect(cvRound(rect.x      * scaleFactor),
          cvRound(rect.y      * scaleFactor),
          cvRound(rect.width  * scaleFactor),
          cvRound(rect.height * scaleFactor)) );
      }
    }
  }
  if (group_rect)
  {
    groupRectangles(layerRect, min_neighbours, mergeRectThreshold);
  }
  rects.insert(rects.end(), layerRect.begin(), layerRect.end());
}

void Detector::DetectMultiScale(const Mat &img, vector<int> &labels,
        vector<double> &scores, vector<Rect> &rects,
        const float detectorThreshold, 
        const double mergeRectThreshold)
{
    CV_Assert(scale > 1.0 && scale <= 2.0);

    vector<Mat> imgPyramid;
    vector<float> scales;
    CreateImagePyramid(img, imgPyramid, scales);

    //for every layer of pyramid
    for (uint i = 0; i < imgPyramid.size(); i++)
    {
        Detect(imgPyramid[i], labels, scores, rects, scales[i], 
          detectorThreshold, mergeRectThreshold);
    }
}
