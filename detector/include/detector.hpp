#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include "classifier.hpp"

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>

class Detector
{
public:
    Detector(std::shared_ptr<Classifier> classifier, cv::Size window_size,
             int dx, int dy, double scale,
             int min_neighbours, bool group_rect);
    void Detect(const cv::Mat &img, std::vector<int> &labels,
                std::vector<double> &scores, std::vector<cv::Rect> &rects,
                const float detectorThreshold = 0.5f);
private:
    void Preprocessing(cv::Mat &img);

    std::shared_ptr<Classifier> classifier;
    cv::Size window_size;
    int dx;
    int dy;
    double scale;
    int min_neighbours;
    bool group_rect;
};

#endif
