#ifndef _TORCH_CLASSIFIER_HPP_
#define _TORCH_CLASSIFIER_HPP_

#include "classifier.hpp"

class TorchClassifier : public Classifier {
public:
    TorchClassifier();
    virtual void SetParams(const std::string& params_string) {}
    virtual void SetParams(const cv::FileNode& params_file_node) {}
    virtual void Init() {}
    virtual Result Classify(cv::Mat& img);
    virtual ~TorchClassifier();
};

#endif