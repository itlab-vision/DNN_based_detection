#ifndef _CLASSIFIER_FACTORY_HPP_
#define _CLASSIFIER_FACTORY_HPP_

#include <memory>

#include "classifier.hpp"

enum ClassifierType
{
    FAKE_CLASSIFIER = 0,
    CAFFE_CLASSIFIER,
    TORCH_CLASSIFIER
};

class ClassifierFactory
{
public:
    ClassifierFactory() { };
    std::shared_ptr<Classifier> CreateClassifier(ClassifierType);
};

#endif
