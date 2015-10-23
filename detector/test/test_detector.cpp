#include "detector.hpp"
#include "classifier_factory.hpp"

#include <gtest.h>
#include <opencv2/core/core.hpp>

#include <vector>
#include <memory>

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)

class TestableDetector : public Detector
{
public:
    TestableDetector(std::shared_ptr<Classifier> classifier,
            cv::Size window_size,
            int dx, int dy, double scale,
            int min_neighbours, bool group_rect) :
        Detector(classifier, window_size, dx, dy, scale,
            min_neighbours, group_rect)
    { }
    using Detector::GetLayerWindowsNumber;
    using Detector::CreateParallelExecutionSchedule;
};

TEST(Detector, check_correctness_number_of_wins_for_empty_img_pyramid)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size window_size(3, 3);
    double scale = 1.2;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    TestableDetector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);
    
    std::vector<int> winNums;
    std::vector<cv::Mat> imgPyramid;
    detector.GetLayerWindowsNumber(imgPyramid, winNums);
    
    EXPECT_EQ(0, winNums.size());
}

TEST(Detector, check_correctness_number_of_wins_for_non_empty_img_pyramid)
{
    ClassifierFactory factory;
    std::shared_ptr<Classifier> classifier = factory.CreateClassifier(FAKE_CLASSIFIER);
    cv::Size window_size(3, 3);
    double scale = 1.2;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    TestableDetector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);
    
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
    cv::Size window_size(3, 3);
    double scale = 1.2;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    TestableDetector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);
    
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
    cv::Size window_size(3, 3);
    double scale = 1.2;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    TestableDetector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);

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
    cv::Size window_size(3, 3);
    double scale = 1.2;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    TestableDetector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);

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
    cv::Size window_size(227, 227);
    double scale = 1.3;
    int min_neighbours = 3, dx = 1, dy = 1;
    bool group_rect = true;
    Detector detector(classifier, window_size, dx, dy,
        scale, min_neighbours, group_rect);

    cv::Mat img(363, 450, CV_8UC3);
    std::vector<cv::Mat> imgPyramid;
    std::vector<float> scales;
    detector.CreateImagePyramid(img, imgPyramid, scales);
    
    EXPECT_EQ(2, imgPyramid.size());
    EXPECT_EQ(2, scales.size());
}
