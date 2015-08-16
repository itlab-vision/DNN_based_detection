#include "Classifier.hpp"

class LuaClassifier : public Classifier {
public:
    LuaClassifier();
    virtual void SetParams(const std::string& params_string) {}
    virtual void SetParams(const cv::FileNode& params_file_node) {}
    virtual void Init(const std::string& file) {}
    virtual Result Classify(cv::Mat& img);
    virtual ~LuaClassifier();
};
