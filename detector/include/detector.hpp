#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include "classifier.hpp"

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>

// TODO: replace it as a class field
const float DETECTOR_THRESHOLD = 0.5f;

class Detector
{
public:
    Detector(std::shared_ptr<Classifier> classifier, cv::Size window_size,
             int dx, int dy, double scale,
             int min_neighbours, bool group_rect);
    void Detect(const cv::Mat &img, std::vector<int> &labels,
                std::vector<double> &scores, std::vector<cv::Rect> &rects);
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
