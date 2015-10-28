#include "caffe_classifier.hpp"

#include "caffe/caffe.hpp"

#include <memory>

using cv::FileNode;
using cv::FileStorage;
using cv::Mat;
using std::shared_ptr;
using std::string;
using std::vector;

struct CaffeClassifier::Impl {
    typedef caffe::Blob<float> Blobf;
    typedef CaffeClassifier::Result Result;

    Impl();
    void FillBlob(const Mat& image,
                  Blobf* blob);
    void FillBlob(const vector<Mat>& images,
                  Blobf* blob);
    vector<Result> GetPrediction(const Blobf* blob);
    void Load();
    void SetParams(const string& params_string);
    void SetParams(const FileNode& params_file_node);

    shared_ptr<caffe::Net<float> > net;
    Blobf* data_blob;
    Blobf* softmax_blob;

    // Computing device to use. Negative values stand for CPU,
    // positives reference GPUs. To learn ID of a particular GPU
    // device_query (either from Caffe or CUDA samples) binary can be used.
    int device_id;

    string net_description_file;
    string net_binary_file;
    string output_blob_name;
};

CaffeClassifier::Impl::Impl()
    : device_id(-1),
      net_description_file(""),
      net_binary_file(""),
      output_blob_name("")
{}

void CaffeClassifier::Impl::SetParams(const string& params_string)
{
    FileStorage fs(params_string,
                   FileStorage::READ + FileStorage::MEMORY);
    SetParams(fs.root());
}

void CaffeClassifier::Impl::SetParams(const FileNode& params_file_node)
{
    CV_Assert(!params_file_node.isNone());
    params_file_node["device_id"] >> device_id;
    params_file_node["net_description_file"] >> net_description_file;
    params_file_node["net_binary_file"] >> net_binary_file;
    params_file_node["output_blob_name"] >> output_blob_name;
}

void CaffeClassifier::Impl::Load()
{
    // Init Caffe.
    using caffe::Caffe;
    if (device_id < 0) {
        Caffe::set_mode(Caffe::CPU);
    } else {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    // Load net decription from a prototxt file.
    net.reset(new caffe::Net<float>(net_description_file, caffe::TEST));
    // Load pre-trained net (binary proto).
    net->CopyTrainedLayersFrom(net_binary_file);

    // Get input blob.
    auto input_blobs = net->input_blobs();
    CV_Assert(input_blobs.size() == 1);
    data_blob = input_blobs[0];

    // Get output blob.
    softmax_blob = net->blob_by_name(output_blob_name).get();
}

void CaffeClassifier::Impl::FillBlob(const Mat& image,
                                     Blobf* blob)
{
    vector<Mat> images;
    images.push_back(image);
    FillBlob(images, blob);
}

void CaffeClassifier::Impl::FillBlob(const vector<Mat>& images,
                                     Blobf* blob)
{
    // Check that net is configured to use a proper batch size.
    CV_Assert(static_cast<size_t>(data_blob->shape(0)) == images.size());
    float* blob_data = blob->mutable_cpu_data();
    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat image = images[i];
        // Check that all other dimentions of blob and image match.
        CV_Assert(blob->shape(1) == image.channels());
        CV_Assert(blob->shape(2) == image.rows);
        CV_Assert(blob->shape(3) == image.cols);

        Mat image_float = image;
        if (image.type() != CV_32F) {
            image.convertTo(image_float, CV_32F);
        }

        vector<Mat> image_channels;
        for (int j = 0; j < image.channels(); ++j)
        {
            image_channels.push_back(Mat(image.size(), CV_32F,
                                         blob_data + blob->offset(i, j)));
        }
        cv::split(image_float, image_channels);
    }
}

vector<CaffeClassifier::Result> CaffeClassifier::Impl::GetPrediction(const Blobf* blob)
{
    // Check that its a binary classifier,
    // i.e. output softmax blob has 2 channels.
    CV_Assert(blob->shape(1) == 2);
    const float* softmax_scores_data = blob->cpu_data();
    vector<Result> results(blob->shape(0));
    for (size_t i = 0; i < results.size(); ++i)
    {
        Result& result = results[i];
        result.confidence = softmax_scores_data[blob->offset(i, 0)];
        result.confidence2 = softmax_scores_data[blob->offset(i, 1)];
        result.label = (result.confidence < result.confidence2) ? 1 : 0;
    }
    return results;
}

CaffeClassifier::CaffeClassifier()
    : impl(new Impl)
{}

void CaffeClassifier::SetParams(const string& params_string)
{
    impl->SetParams(params_string);
}

void CaffeClassifier::SetParams(const FileNode& params_file_node)
{
    impl->SetParams(params_file_node);
}

void CaffeClassifier::Init()
{
    impl->Load();
}

CaffeClassifier::Result CaffeClassifier::Classify(Mat& image)
{
    // Write image to the input blob of the net.
    impl->FillBlob(image, impl->data_blob);
    // Pass data through the net.
    impl->net->ForwardPrefilled();
    // Get classification result.
    return impl->GetPrediction(impl->softmax_blob)[0];
}

vector<CaffeClassifier::Result> CaffeClassifier::Classify(const vector<Mat>& images)
{
    impl->FillBlob(images, impl->data_blob);
    // Pass data through the net.
    impl->net->ForwardPrefilled();
    // Get classification result.
    return impl->GetPrediction(impl->softmax_blob);
}

CaffeClassifier::~CaffeClassifier()
{}

