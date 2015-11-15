#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include "classifier.hpp"

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>

class Detector
{
public:
    Detector(std::shared_ptr<Classifier> classifier,
             cv::Size max_window_size, cv::Size min_window_size,
             int kPyramidLevels, int dx, int dy,
             int min_neighbours, bool group_rect,bool nms_max,bool nms_avg);
    void Detect(cv::Mat &layer, std::vector<int> &labels,
            std::vector<double> &scores, std::vector<cv::Rect> &rects,
            const float scaleFactor,
            const float detectorThreshold, 
            const double mergeRectThreshold);
    void DetectMultiScale(const cv::Mat &img, std::vector<int> &labels,
            std::vector<double> &scores, std::vector<cv::Rect> &rects,
            const float detectorThreshold = 0.5f,
            const double mergeRectThreshold = 0.2);
    void CreateImagePyramid(const cv::Mat &img, std::vector<cv::Mat> &pyramid,
                            std::vector<float> &scales);
    void NMS_max(std::vector<int> &labels, std::vector<double> &scores, std::vector<cv::Rect> &rects,
            const double theshold_overlap = 0.6); 
    void NMS_avg(std::vector<int> &labels, std::vector<double> &scores, std::vector<cv::Rect> &rects, 
            const double mergeRectThreshold = 0.8);

protected:
    void Preprocessing(cv::Mat &img);
#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
    void GetLayerWindowsNumber(std::vector<cv::Mat> &imgPyramid,
        std::vector<int> &winNum);
    void CreateParallelExecutionSchedule(std::vector<int> &winNum,
        std::vector<std::vector<int> > &levels);    
    void Detect(std::vector<cv::Mat> &imgPyramid,
            std::vector<std::vector<int> > &levels,
            std::vector<float> &scales,
            std::vector<int> &labels,
            std::vector<double> &scores, std::vector<cv::Rect> &rects,
            const float detectorThreshold = 0.5f,
            const double mergeRectThreshold = 0.2);
#endif

    std::shared_ptr<Classifier> classifier;
    cv::Size max_window_size;
    cv::Size min_window_size;
    int kPyramidLevels;
    int dx;
    int dy;
    int min_neighbours;
    bool group_rect;
    bool nms_max;
    bool nms_avg;
};

#endif
