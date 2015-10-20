#ifndef _TORCH_CLASSIFIER_HPP_
#define _TORCH_CLASSIFIER_HPP_

#include "classifier.hpp"

class TorchClassifier : public Classifier {
public:
    typedef Classifier::Result Result;

    virtual void SetParams(const std::string& params_string) {}
    virtual void SetParams(const cv::FileNode& params_file_node) {}
    virtual void Init() {}
    virtual Result Classify(cv::Mat& img) { return Result(); }
    virtual ~TorchClassifier() { };

    friend ClassifierFactory;

protected:
    TorchClassifier() { };
};

#endif