#include "detector.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
#include <mpi.h>
#define MAX_BBOXES_NUMBER 1000
#endif

using namespace cv;
using namespace std;

void Detector::Preprocessing(Mat &img)
{
    float mean[] = {0.40559885502486, -0.019621851500929, 0.026953143125972};
    float std[] = {0.26126178026709, 0.049694558439293, 0.071862255292542};
    img.convertTo(img, CV_32F, 1.0f/255.0f);
    cvtColor(img, img, COLOR_BGR2YCrCb);

    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            Vec3f pixel = img.at<Vec3f>(y, x);
            for (int z = 0; z < 3; z++)
            {
                pixel[z] = (pixel[z] - mean[z]) / std[z];
            }
        }
    }
}

Detector::Detector(std::shared_ptr<Classifier> classifier_,
                   cv::Size window_size_, int dx_ = 1, int dy_ = 1,
                   double scale_ = 1.2, int min_neighbours_ = 3,
                   bool group_rect_ = false)
    : classifier(classifier_),
      window_size(window_size_),
      dx(dx_),
      dy(dy_),
      scale(scale_),
      min_neighbours(min_neighbours_),
      group_rect(group_rect_)
{}

void Detector::CreateImagePyramid(const cv::Mat &img, std::vector<Mat> &pyramid,
                                  std::vector<float> &scales)
{
    pyramid.clear();    
    Mat resizedImg;
    img.copyTo(resizedImg);
    float scaleFactor = 1.0f;
    // FIX: need to increase image size to detect small faces
    while (resizedImg.cols >= window_size.width &&
           resizedImg.rows >= window_size.height)
    {
        pyramid.push_back(resizedImg.clone());
        scales.push_back(scaleFactor);
        scaleFactor *= scale;
        resize(img, resizedImg,
               Size((int)(img.cols / scaleFactor), (int)(img.rows / scaleFactor)),
               0, 0, INTER_LINEAR);
    }
}

void Detector::Detect(Mat &layer, vector<int> &labels,
        vector<double> &scores, vector<Rect> &rects,
        const float scaleFactor,
        const float detectorThreshold, 
        const double mergeRectThreshold)
{
    vector<Rect> layerRect;
    for (int y = 0; y < layer.rows - window_size.height + 1; y += dy)
    {
        for (int x = 0; x < layer.cols - window_size.width + 1; x += dx)
        {
            Rect rect(x, y, window_size.width, window_size.height);
            Mat window = layer(rect);

            Classifier::Result result = classifier->Classify(window);
            // FIX: class label - 0
            if (fabs(result.confidence) > detectorThreshold && result.label == 0)
            {
                labels.push_back(result.label);
                scores.push_back(result.confidence);
                layerRect.push_back(
                    Rect(cvRound(rect.x      * scaleFactor),
                         cvRound(rect.y      * scaleFactor),
                         cvRound(rect.width  * scaleFactor),
                         cvRound(rect.height * scaleFactor)) );
            }
        }
    }
    if (group_rect)
    {
        // FIX: groupRectangkes doesn't modidficate labels and scores
        groupRectangles(layerRect, min_neighbours, mergeRectThreshold);
    }
    rects.insert(rects.end(), layerRect.begin(), layerRect.end());
}

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
void Detector::GetLayerWindowsNumber(std::vector<cv::Mat> &imgPyramid,
        std::vector<int> &winNum)
{
    winNum.clear();
    int kLayers = imgPyramid.size();
    for (int i = 0; i < kLayers; i++)
    {       
        int kWins = (imgPyramid[i].cols - window_size.width + 1) * 
                    (imgPyramid[i].rows - window_size.height + 1) / 
                    (dx * dy);
        winNum.push_back(kWins);
    }
}

void Detector::CreateParallelExecutionSchedule(std::vector<int> &winNum,
        std::vector<std::vector<int> > &levels)
{
    // sort by descending order
    int kLevels = winNum.size(), np = levels.size();
    vector<int> indeces(kLevels), weights(np), disp(np);
    for (int i = 0; i < kLevels; i++)
    {
        indeces[i] = i;
    }
    sort(indeces.begin(), indeces.end(),
      [&winNum](size_t i1, size_t i2) 
        { return winNum[i1] > winNum[i2]; });
    sort(winNum.begin(), winNum.end(), 
      [](int a, int b) 
        { return a > b; });
    // bigest layers will be processed by different processes
    if (kLevels <= np)
    {
        for (int i = 0; i < kLevels; i++)
        {
            levels[i].push_back(indeces[i]);
        }
        return;
    }
    for (int i = 0; i < np; i++)
    {
        levels[i].push_back(indeces[i]);
        weights[i] = winNum[i];
        disp[i] = 0;
    }
    // distribute other layers
    for (int i = np; i < kLevels; ++i)
    {
        for (int j = 0; j < np; j++)
        {
            // try to add the next layer to process j and compute variance
            weights[j] += winNum[i];
            int minValue = weights[0], maxValue = weights[0];
            for (int k = 1; k < np; k++)
            {
                minValue = min(minValue, weights[k]);
                maxValue = max(maxValue, weights[k]);
            }
            disp[j] = maxValue - minValue;
            weights[j] -= winNum[i];
        }
        // choose the process provided minimum variance
        int minValue = disp[0], argMin = 0;
        for (int j = 1; j < np; j++)
        {
            if (disp[j] < minValue)
            {
                minValue = disp[j];
                argMin = j;
            }
        }
        // add index of the layer to the correspond process
        levels[argMin].push_back(indeces[i]);
    }    
}

void Detector::Detect(std::vector<cv::Mat> &imgPyramid,
        vector<vector<int> > &levels, vector<float> &scales,
        std::vector<int> &labels, std::vector<double> &scores,
        std::vector<cv::Rect> &rects,
        const float detectorThreshold,
        const double mergeRectThreshold)
{
    // process levels set to the particular process
    int np, rank;
    np = levels.size();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    vector<int> procLevels = levels[rank];
    int kLevels = procLevels.size();
    vector<int> procLabels;
    vector<double> procScores;
    vector<Rect> procRects;    
    for (int i = 0; i < kLevels; i++)
    {
        int levelId = procLevels[i];        
        Detect(imgPyramid[levelId], procLabels, procScores, procRects,
            scales[levelId], detectorThreshold, mergeRectThreshold);
        cout <<  "Process " << rank << ": " << endl 
             << "\tLevelId: " <<  levelId << " (scale = " << scales[levelId]
             << "), kLabels = " << procLabels.size()
             << ", kScores = " << procScores.size()
             << ", kRects =  " << procRects.size() << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // recieve results to process 0
    if (rank == 0)
    {
        labels.insert(labels.end(), procLabels.begin(), procLabels.end());
        scores.insert(scores.end(), procScores.begin(), procScores.end());
        rects.insert(rects.end(), procRects.begin(), procRects.end());
        for (int i = 1; i < np; i++)
        {
            int bboxesNum = 0;
            vector<int> childProcLabels(MAX_BBOXES_NUMBER);
            MPI_Status status;
            MPI_Recv(childProcLabels.data(), MAX_BBOXES_NUMBER, MPI_INT,
                i, 1, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &bboxesNum);
            childProcLabels.resize(bboxesNum);

            bboxesNum = 0;
            vector<double> childProcScores(MAX_BBOXES_NUMBER);
            MPI_Recv(childProcScores.data(), MAX_BBOXES_NUMBER, MPI_DOUBLE,
                i, 2, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &bboxesNum);
            childProcScores.resize(bboxesNum);
            
            bboxesNum = 0;
            vector<Rect> childProcRects(MAX_BBOXES_NUMBER);
            MPI_Recv(childProcRects.data(), MAX_BBOXES_NUMBER * sizeof(Rect) / sizeof(int),
                MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &bboxesNum);
            childProcLabels.resize(bboxesNum / 4);

            bboxesNum /= 4;
            labels.insert(labels.end(), childProcLabels.begin(), childProcLabels.begin() + bboxesNum);
            scores.insert(scores.end(), childProcScores.begin(), childProcScores.begin() + bboxesNum);
            rects.insert(rects.end(), childProcRects.begin(), childProcRects.begin() + bboxesNum);            
        }        
    }
    else
    {        
        MPI_Send(procLabels.data(), procRects.size(), MPI_INT, 0, 1, MPI_COMM_WORLD);
        
        MPI_Send(procScores.data(), procRects.size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);        
        
        MPI_Send(procRects.data(), procRects.size() * sizeof(Rect) / sizeof(int), 
            MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
}
#endif

void Detector::DetectMultiScale(const Mat &img, vector<int> &labels,
        vector<double> &scores, vector<Rect> &rects,
        const float detectorThreshold, 
        const double mergeRectThreshold)
{
    CV_Assert(scale > 1.0 && scale <= 2.0); 

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // 1. create scale pyramid
    vector<Mat> imgPyramid;
    vector<float> scales;
    CreateImagePyramid(img, imgPyramid, scales);
    // 2. compute number of windows at each layer
    vector<int> winNum;
    GetLayerWindowsNumber(imgPyramid, winNum);
    // 3. create schedule to send layers
    vector<vector<int> > levels(np);
    CreateParallelExecutionSchedule(winNum, levels);
    // 4. send layers to child processes and detect  objects on the first layer
    Detect(imgPyramid, levels, scales, labels, scores, rects,
      detectorThreshold, mergeRectThreshold);
#else
    vector<Mat> imgPyramid;
    vector<float> scales;
    CreateImagePyramid(img, imgPyramid, scales);
    for (uint i = 0; i < imgPyramid.size(); i++)
    {
        Detect(imgPyramid[i], labels, scores, rects, scales[i], 
          detectorThreshold, mergeRectThreshold);
    }
#endif
}
