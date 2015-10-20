#include "classifier_factory.hpp"
#include "fake_classifier.hpp"

#ifdef HAVE_CAFFE
    #include "caffe_classifier.hpp"
#endif

#ifdef HAVE_TORCH
    #include "torch_classifier.hpp"
#endif


std::shared_ptr<Classifier> ClassifierFactory::CreateClassifier(ClassifierType type)
{
    switch (type)
    {
    case CAFFE_CLASSIFIER:
        {
#ifdef HAVE_CAFFE
            std::shared_ptr<Classifier> classifier(new CaffeClassifier());
            return classifier;
#endif
            break;
        }   
    case TORCH_CLASSIFIER:
        {
#ifdef HAVE_TORCH
            std::shared_ptr<Classifier> classifier(new TorchClassifier());
            return classifier;
#endif
            break;
        }
        default:
        {
            std::shared_ptr<Classifier> classifier(new FakeClassifier());
            return classifier;
        }
    }
    std::shared_ptr<Classifier> classifier(new FakeClassifier());
    return classifier;
}