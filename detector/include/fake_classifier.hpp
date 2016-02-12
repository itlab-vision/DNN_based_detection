#ifndef _FAKE_CLASSIFIER_HPP_
#define _FAKE_CLASSIFIER_HPP_

#include <exception>

#include "classifier.hpp"

class NotImplemented : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Class is not implemented.";
    }
};

class FakeClassifier : public Classifier
{
public:
    typedef Classifier::Result Result;
    
    virtual void SetParams(const std::string& params_string)
    { 
        throw NotImplemented();
    }
    virtual void SetParams(const cv::FileNode& params_file_node)
    { 
        throw NotImplemented();
    }
    virtual void Init()
    { 
        throw NotImplemented();
    }
    virtual Result Classify(cv::Mat &img)
    { 
        throw NotImplemented();
    }
    virtual std::vector<Result> Classify(const std::vector<cv::Mat>& images)
    {
        throw NotImplemented();
    }
    virtual ~FakeClassifier() { }

    friend class ClassifierFactory;

protected:
    FakeClassifier() 
    { }
};

#endif