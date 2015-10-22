#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "detector.hpp"
#include "classifier_factory.hpp"

#if defined(HAVE_MPI) && defined(PAR_SET_IMAGES)
#include <mpi.h>
#endif

using namespace std;
using cv::Rect;
using cv::Mat;
using cv::Size;
using cv::FileNode;
using cv::FileStorage;

struct Args {
    string input_path;
    FileNode params_file_node;
    vector<string> filenames;
};

const char* help = "detector \"input/folder\"\n\
    input folder must contain two files:\n\
    annotation.txt - contains filenames of imgs to detect\n\
    config.yml - contains description of classifier:\n\
        step (int)\n\
        min_neighbs (int)\n\
        scale (float)\n\
        group_rect (int 0 or 1)\n\
        device_id - <0 cpu, >=0 gpu\n\
        net_description_file\n\
        net_binary_file\n\
    output of the program is file result.txt in input folder\n\
        \n";

void detect(Detector &detector, std::string &fileName, FILE *out)
{
    Mat img = imread(fileName, cv::IMREAD_COLOR);
    cout << "Processing " << fileName << endl;
    
    vector<int> labels;
    vector<Rect> rects;
    vector<double> scores;
    detector.DetectMultiScale(img, labels, scores, rects);

    string fileLine = fileName + "\n" + to_string(rects.size()) + "\n";
    for (size_t j = 0; j < rects.size(); j++)
    {
        fileLine += to_string(rects[j].x) + " " + to_string(rects[j].y) + " "
            + to_string(rects[j].width) + " " + to_string(rects[j].height) + " "
            + to_string(scores[j]) + " \n";
    }
    fprintf(out, "%s", fileLine.c_str());
}

#if defined(HAVE_MPI) && defined(PAR_SET_IMAGES)
void detect(shared_ptr<Classifier> classifier, Args args, FILE *out) 
{
    int argc, rank, np, fileStep, leftIdx, rigthIdx;
    char **argv;
    MPI_Init(&argc, &argv);

    FileNode params = args.params_file_node;
    int step = params["step"];
    float scale = params["scale"];
    int min_neighbs = params["min_neighbs"];
    int group_rect = params["group_rect"];

    Detector detector(classifier, Size(227, 227),
                      step, step, scale, min_neighbs, group_rect);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fileStep = args.filenames.size() / np;
    leftIdx = rank * fileStep;
    rigthIdx = min(rank * (fileStep + 1), (int)(args.filenames.size()));
    for (int i = leftIdx; i < rigthIdx; i++)
    {
        detect(detector, args.filenames[i], out);
    }

    MPI_Finalize();
}    
#else
void detect(shared_ptr<Classifier> classifier, Args args, FILE *out)
{
    FileNode params = args.params_file_node;
    int step = params["step"];
    float scale = params["scale"];
    int min_neighbs = params["min_neighbs"];
    int group_rect = params["group_rect"];

    Detector detector(classifier, Size(227, 227),
                      step, step, scale, min_neighbs, group_rect);

    for (size_t i = 0; i < args.filenames.size(); i++)
    {
        detect(detector, args.filenames[i], out);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Too few arguments\n" << help;
        return 1;
    }

    Args args;
    args.input_path = string(argv[1]);

    string annot = args.input_path + "/annotation.txt";
    string config = args.input_path + "/config.yml";

    ifstream content_annot(annot);
    FileStorage fs(config, FileStorage::READ);

    if (!content_annot.is_open() || !fs.isOpened()) {
        cout << "Cannot find or open input files\n";
        cout << help;
        return 1;
    }

    std::string s;
    while (std::getline(content_annot, s)) {
        args.filenames.push_back(s);
    }

    args.params_file_node = fs.root();

    ClassifierFactory factory;
    shared_ptr<Classifier> classifier = factory.CreateClassifier(CAFFE_CLASSIFIER);
    classifier->SetParams(args.params_file_node);
    classifier->Init();

    string fileName = args.input_path + "/result.txt";
    FILE *out = fopen(fileName.c_str(), "w");
    if (out == NULL) {
        cout << "Problems with creating output file\n";
        cout << help;
        return 1;
    }

    detect(classifier, args, out);

    fclose(out);
    return 0;
}
