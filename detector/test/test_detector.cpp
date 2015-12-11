#include "detector.hpp"
#include "classifier_factory.hpp"

#include <gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <memory>

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)

class TestableDetector : public Detector
{
public:
    TestableDetector(std::shared_ptr<Classifier> classifier,
             cv::Size max_window_size, cv::Size min_window_size,
             int kPyramidLevels, int dx = 1, int dy = 1,
             int min_neighbours = 3, NMS_TYPE nms_type = NMS_NONE) :
        Detector(classifier, max_window_size, min_window_size, kPyramidLevels,
            dx, dy, min_neighbours, nms_type)
    { }
    using Detector::GetLayerWindowsNumber;
    using Detector::CreateParallelExecutionSchedule;
};

TEST(Detector, check_correctness_number_of_wins_for_empty_img_pyramid)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(3, 3), min_window_size(2, 2);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_NONE;
    TestableDetector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);
    
    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    detector.GetLayerWindowsNumber(imgPyramid, winNums);
    
    EXPECT_EQ(0, winNums.size());
}

TEST(Detector, check_correctness_number_of_wins_for_non_empty_img_pyramid)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(3, 3), min_window_size(2, 2);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_NONE;
    TestableDetector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);
    
    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    for (int i = 3; i < 10; ++i)
    {
        imgPyramid.push_back(cv::Mat::zeros(i, i, CV_8UC1));
    }
    detector.GetLayerWindowsNumber(imgPyramid, winNums);
    
    EXPECT_EQ(7, winNums.size());
}

TEST(Detector, check_number_of_scales_in_schedule_when_nscales_less_than_np)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(3, 3), min_window_size(2, 2);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_NONE;
    TestableDetector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);
    
    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    for (int i = 3; i < 5; ++i)
    {
        imgPyramid.push_back(cv::Mat::zeros(i, i, CV_8UC1));
    }
    detector.GetLayerWindowsNumber(imgPyramid, winNums);

    std::vector<std::vector<int> > levels(4);
    detector.CreateParallelExecutionSchedule(winNums, levels);

    EXPECT_EQ(1, levels[0].size());
    EXPECT_EQ(1, levels[1].size());
    EXPECT_EQ(0, levels[2].size());
    EXPECT_EQ(0, levels[3].size());
}

TEST(Detector, check_number_of_scales_in_schedule_when_nscales_more_than_np)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(3, 3), min_window_size(2, 2);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_NONE;
    TestableDetector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);

    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    for (int i = 3; i < 10; ++i)
    {
        imgPyramid.push_back(cv::Mat::zeros(i, i, CV_8UC1));
    }
    detector.GetLayerWindowsNumber(imgPyramid, winNums);

    std::vector<std::vector<int> > levels(4);
    detector.CreateParallelExecutionSchedule(winNums, levels);

    EXPECT_EQ(1, levels[0].size());
    EXPECT_EQ(1, levels[1].size());
    EXPECT_EQ(1, levels[2].size());
    EXPECT_EQ(4, levels[3].size());
}

TEST(Detector, check_correctness_of_schedule_when_nscales_more_than_np)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(3, 3), min_window_size(2, 2);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nmsType = NMS_NONE;
    TestableDetector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, group_rect);

    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    for (int i = 3; i < 10; ++i)
    {
        imgPyramid.push_back(cv::Mat::zeros(i, i, CV_8UC1));
    }
    detector.GetLayerWindowsNumber(imgPyramid, winNums);

    std::vector<std::vector<int> > levels(4);
    detector.CreateParallelExecutionSchedule(winNums, levels);

    EXPECT_EQ(6, levels[0][0]);
    EXPECT_EQ(5, levels[1][0]);
    EXPECT_EQ(4, levels[2][0]);
    EXPECT_EQ(3, levels[3][0]);
    EXPECT_EQ(2, levels[3][1]);
    EXPECT_EQ(1, levels[3][2]);
    EXPECT_EQ(0, levels[3][3]);
}

#endif

TEST(Detector, check_win_num_min_stride)
{
    int kCols = 10, kRows = 10, winWidth = 3, winHeight = 3, dx = 1, dy = 1;
    int kWins = (kCols - winWidth + 1) * (kRows - winHeight + 1) / (dx * dy);
    EXPECT_EQ(64, kWins);
}

TEST(Detector, check_win_num)
{
    int kCols = 10, kRows = 10, winWidth = 3, winHeight = 3, dx = 2, dy = 2;
    int kWins = (kCols - winWidth + 1) * (kRows - winHeight + 1) / (dx * dy);
    EXPECT_EQ(16, kWins);
}

TEST(Detector, check_number_of_levels_in_image_pyramid)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(227, 227), min_window_size(60, 60);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_NONE;
    Detector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);

    cv::Mat img(363, 450, CV_8UC3);
    std::vector<cv::Mat> imgPyramid;
    std::vector<float> scales;
    detector.CreateImagePyramid(img, imgPyramid, scales);
    
    EXPECT_EQ(2, imgPyramid.size());
    EXPECT_EQ(2, scales.size());
}

TEST(Detector, check_create_image_pyramid_failed)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(227, 227), min_window_size(60, 60);
    int kPyramidLevels = 3;
    int min_neighbours = 3, dx = 1, dy = 1;
    NMS_TYPE nms_type = NMS_NONE;
    Detector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);

    cv::Mat img(363, 450, CV_8UC3);
    std::vector<cv::Mat> imgPyramid;
    std::vector<float> scales;
    detector.CreateImagePyramid(img, imgPyramid, scales);
    for (int i = 0; i < kPyramidLevels; i++)
    {
        std::cout << imgPyramid[i].rows << "\t"
                  << imgPyramid[i].cols << "\t"
                  << scales[i] << std::endl;
    }
}

TEST(Detector, check_image_pyramid_algorithm)
{
    cv::Size maxWinSize(227, 227), minWinSize(60, 60);
    int rows = 363, cols = 450;
    int kPyramidLevels = 11;

    std::vector<cv::Mat> pyramid;
    std::vector<float> scales;    
    int kLevels = 0;
    float scale = powf(((float)maxWinSize.width) / ((float)minWinSize.width), 
                       1.0f / ((float)kPyramidLevels - 1.0f));
    std::cout << "scale = " << scale << std::endl;
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC3), resizedImg;

    img.copyTo(resizedImg);
    float scaleFactor = 1.0f;
    // decrease image size = increase window size
    while (resizedImg.cols >= maxWinSize.width &&
           resizedImg.rows >= maxWinSize.height)
    {
        pyramid.push_back(resizedImg.clone());
        scales.push_back(scaleFactor);
        scaleFactor /= scale;        
        cv::resize(img, resizedImg,
               cv::Size((int)(img.cols * scaleFactor), (int)(img.rows * scaleFactor)),
               0, 0, cv::INTER_LINEAR);
        kLevels++;
    }
    // increase image size = decrease window size
    scaleFactor = 1.0f;
    while (kLevels < kPyramidLevels)
    {
        scaleFactor *= scale;
        cv::resize(img, resizedImg,
               cv::Size((int)(img.cols * scaleFactor), (int)(img.rows * scaleFactor)),
               0, 0, cv::INTER_LINEAR);
        pyramid.push_back(resizedImg.clone());
        scales.push_back(scaleFactor);
        kLevels++;
    }    

    for (int i = 0; i < kLevels; i++)
    {
        std::cout << pyramid[i].rows << "\t"
                  << pyramid[i].cols << "\t"
                  << scales[i] << std::endl;
    }
}

TEST(Detector, check_scores_sorting)
{
    std::vector<cv::Rect> rects;
    std::vector<int> labels;
    std::vector<double> scores;
    
    rects.push_back(cv::Rect(1, 1, 3, 6));
    rects.push_back(cv::Rect(1, 3, 5, 7));
    rects.push_back(cv::Rect(0, 1, 4, 6));
    rects.push_back(cv::Rect(2, 1, 6, 9));
    
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    scores.push_back(-0.4);
    scores.push_back(0.4);
    scores.push_back(0.8);
    scores.push_back(0.36);

    int kRects = rects.size();
    for (int i = 0; i < kRects; i++)
    {
        for (int j = kRects - 1; j >= i + 1; j--)
        {
            if (scores[j] > scores[j - 1])
            {
                double score = scores[j];
                scores[j] = scores[j - 1];
                scores[j - 1] = score;

                int label = labels[j];
                labels[j] = labels[j - 1];
                labels[j - 1] = label;

                cv::Rect rect = rects[j];
                rects[j] = rects[j - 1];
                rects[j - 1] = rect;
            }
        }
    }

    EXPECT_EQ(cv::Rect(0, 1, 4, 6), rects[0]);
    EXPECT_EQ(cv::Rect(1, 3, 5, 7), rects[1]);
    EXPECT_EQ(cv::Rect(2, 1, 6, 9), rects[2]);
    EXPECT_EQ(cv::Rect(1, 1, 3, 6), rects[3]);

    EXPECT_EQ(0.8, scores[0]);
    EXPECT_EQ(0.4, scores[1]);
    EXPECT_EQ(0.36, scores[2]);
    EXPECT_EQ(-0.4, scores[3]);
}

TEST(Detector, check_group_rects_by_maximum_score)
{
    std::vector<cv::Rect> rects;
    std::vector<int> labels;
    std::vector<double> scores;
    
    rects.push_back(cv::Rect(1, 1, 3, 6));
    rects.push_back(cv::Rect(1, 3, 5, 7));
    rects.push_back(cv::Rect(0, 1, 4, 6));
    rects.push_back(cv::Rect(2, 1, 6, 9));
    
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    scores.push_back(-0.4);
    scores.push_back(0.4);
    scores.push_back(0.8);
    scores.push_back(0.36);

    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(227, 227), min_window_size(60, 60);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_MAX;
    Detector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);
    double mergeThreshold = 0.5;
    detector.GroupRectangles(rects, labels, scores, mergeThreshold);
    
    EXPECT_EQ(3, rects.size());
    EXPECT_EQ(cv::Rect(0, 1, 4, 6), rects[0]);
    EXPECT_EQ(cv::Rect(1, 3, 5, 7), rects[1]);
    EXPECT_EQ(cv::Rect(2, 1, 6, 9), rects[2]);
}

TEST(Detector, check_group_rects_by_average)
{
    std::vector<cv::Rect> rects;
    std::vector<int> labels;
    std::vector<double> scores;
    
    rects.push_back(cv::Rect(1, 1, 3, 6));
    rects.push_back(cv::Rect(1, 3, 5, 7));
    rects.push_back(cv::Rect(0, 1, 4, 6));
    rects.push_back(cv::Rect(2, 1, 6, 9));
    
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    scores.push_back(-0.4);
    scores.push_back(0.4);
    scores.push_back(0.8);
    scores.push_back(0.36);

    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size max_window_size(227, 227), min_window_size(60, 60);
    int min_neighbours = 3, dx = 1, dy = 1, kPyramidLevels = 2;
    NMS_TYPE nms_type = NMS_AVG;
    Detector detector(classifier, max_window_size, min_window_size,
        kPyramidLevels, dx, dy, min_neighbours, nms_type);
    double mergeThreshold = 0.5;
    detector.GroupRectangles(rects, labels, scores, mergeThreshold);

    EXPECT_EQ(3, rects.size());
    EXPECT_EQ(cv::Rect(0, 1, 3, 6), rects[0]);
    EXPECT_EQ(cv::Rect(1, 3, 5, 7), rects[1]);
    EXPECT_EQ(cv::Rect(2, 1, 6, 9), rects[2]);
}