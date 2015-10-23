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
