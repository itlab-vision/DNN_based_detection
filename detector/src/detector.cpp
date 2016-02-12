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
    img.convertTo(img, CV_32F, 1.0f / 255.0f);
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
             Size window_size_,
             Size max_window_size_, Size min_window_size_,
             int kPyramidLevels_, int dx_ = 1, int dy_ = 1,
             int min_neighbours_ = 3, bool group_rect_ = false)
    : classifier(classifier_),
      window_size(window_size_),
      max_window_size(max_window_size_),
      min_window_size(min_window_size_),
      kPyramidLevels(kPyramidLevels_),
      dx(dx_),
      dy(dy_),
      min_neighbours(min_neighbours_),
      group_rect(group_rect_)
{}

void Detector::CreateImagePyramid(const Mat &img, vector<Mat> &pyramid,
                                  vector<float> &scales)
{
    CV_Assert(!img.empty());
    pyramid.clear();
    scales.clear();

    float scaleMin = min_window_size.height / static_cast<float>(window_size.height);
    float scaleMax = max_window_size.height / static_cast<float>(window_size.height);
    float scaleStep = powf(scaleMax / scaleMin, 1.0f / (kPyramidLevels - 1.0f));
    float scale = scaleMin;
    for (int i = 0; i < kPyramidLevels; ++i) {
        Mat resizedImg;
        resize(img, resizedImg,
               Size(img.cols / scale, img.rows / scale),
               0, 0, INTER_LINEAR);
        pyramid.push_back(resizedImg.clone());
        scales.push_back(scale);
        scale *= scaleStep;
    }
}


void Detector::Detect(Mat &layer, vector<int> &labels,
        vector<double> &scores, vector<Rect> &rects,
        const float scaleFactor,
        const float detectorThreshold)
{
    int windowsNum = ((layer.cols - window_size.width) / dx + 1) *
                     ((layer.rows - window_size.height) / dy + 1);
    if (windowsNum <= 0)
    {
        return;
    }
    vector<Rect> rois(windowsNum);
    vector<Mat> windows(windowsNum);
    int i = 0;
    for (int y = 0; y < layer.rows - window_size.height + 1; y += dy)
    {
        for (int x = 0; x < layer.cols - window_size.width + 1; x += dx)
        {
            Rect rect(x, y, window_size.width, window_size.height);
            rois[i] = rect;
            windows[i] = layer(rect);
            ++i;
        }
    }

    vector<Classifier::Result> results = classifier->Classify(windows);
    for (int j = 0; j < windowsNum; ++j)
    {
        const Classifier::Result& res = results[j];
        const Rect& r = rois[j];
        if (res.confidence2 > detectorThreshold && res.label != 0)
        {
            labels.push_back(res.label);
            scores.push_back(res.confidence2);
            rects.push_back(
                Rect(cvRound(r.x      * scaleFactor),
                     cvRound(r.y      * scaleFactor),
                     cvRound(r.width  * scaleFactor),
                     cvRound(r.height * scaleFactor)) );
        }
    }
}

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
void Detector::GetLayerWindowsNumber(vector<Mat> &imgPyramid, vector<int> &winNum)
{
    winNum.clear();
    int kLayers = imgPyramid.size();
    for (int i = 0; i < kLayers; i++)
    {       
        int kWins = ((imgPyramid[i].cols - window_size.width) / dx + 1) *
                    ((imgPyramid[i].rows - window_size.height) / dy + 1);
        winNum.push_back(kWins);
    }
}

void Detector::CreateParallelExecutionSchedule(vector<int> &winNum,
        vector<vector<int> > &levels)
{
    // sort in descending order.
    int kLevels = winNum.size(), np = levels.size();
    vector<int> indices(kLevels), weights(np), disp(np);
    for (int i = 0; i < kLevels; i++)
    {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(),
      [&winNum](size_t i1, size_t i2) 
        { return winNum[i1] > winNum[i2]; });
    sort(winNum.begin(), winNum.end(), 
      [](int a, int b) 
        { return a > b; });
    // biggest layers will be processed by different processes.
    if (kLevels <= np)
    {
        for (int i = 0; i < kLevels; i++)
        {
            levels[i].push_back(indices[i]);
        }
        return;
    }
    for (int i = 0; i < np; i++)
    {
        levels[i].push_back(indices[i]);
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
        // add index of the layer to the corresponding process.
        levels[argMin].push_back(indices[i]);
    }    
}

void Detector::Detect(vector<Mat> &imgPyramid,
        vector<vector<int> > &levels, vector<float> &scales,
        vector<int> &labels, vector<double> &scores,
        vector<Rect> &rects,
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
        cout << "Process " << rank << ": " << endl
             << "\tLevelId: " <<  levelId << " (scale = " << scales[levelId] << ")" << endl;
        Detect(imgPyramid[levelId], procLabels, procScores, procRects,
            scales[levelId], detectorThreshold);
        cout << "kLabels = " << procLabels.size()
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
    for (size_t i = 0; i < imgPyramid.size(); i++)
    {
        Detect(imgPyramid[i], labels, scores, rects, scales[i], detectorThreshold);
        if (group_rect)
        {
            // FIX: groupRectangles doesn't modify labels and scores.
            groupRectangles(rects, min_neighbours, mergeRectThreshold);
        }
    }
#endif
}
