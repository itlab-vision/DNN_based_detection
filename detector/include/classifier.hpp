#ifndef _CLASSIFIER_HPP_
#define _CLASSIFIER_HPP_

#include <opencv2/core/core.hpp>

class ClassifierFactory;

class Classifier
{
public:
    struct Result
    {
        int label;
        float confidence;
        float confidence2;
    };

    Classifier() { };
    virtual void SetParams(const std::string& params_string) = 0;
    virtual void SetParams(const cv::FileNode& params_file_node) = 0;
    virtual void Init() = 0;
    virtual Result Classify(cv::Mat &img) = 0;
    virtual std::vector<Result> Classify(const std::vector<cv::Mat>& images) = 0;
    virtual ~Classifier() { };
};

#endif
