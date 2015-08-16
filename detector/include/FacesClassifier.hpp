#include "Classifier.hpp"

class FacesClassifier : public Classifier {
public:
	FacesClassifier();
    virtual void SetParams(const std::string& params_string) {}
    virtual void SetParams(const cv::FileNode& params_file_node) {}
    virtual void Init(const std::string& file) {}
    virtual Result Classify(cv::Mat& img);
    virtual ~FacesClassifier();
};
