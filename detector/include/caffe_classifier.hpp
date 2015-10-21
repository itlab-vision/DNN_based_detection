#ifndef _CAFFE_CLASSIFIER_HPP_
#define _CAFFE_CLASSIFIER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

#include "classifier.hpp"

class CaffeClassifier : public Classifier
{
public:
    typedef Classifier::Result Result;

    virtual void SetParams(const std::string& params_string);
    virtual void SetParams(const cv::FileNode& params_file_node);
    virtual void Init();
    virtual Result Classify(cv::Mat& image);
    virtual std::vector<Result> Classify(const std::vector<cv::Mat>& images);    
    virtual ~CaffeClassifier();

    friend ClassifierFactory;

private:
    struct Impl;
    std::shared_ptr<Impl> impl;
protected:
    CaffeClassifier();
};

#endif
