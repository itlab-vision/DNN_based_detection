#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "detector.hpp"
#include "classifier_factory.hpp"

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

void detect(shared_ptr<Classifier> classifier, Args args, ofstream &out) {
    vector<int> labels;
    vector<Rect> rects;
    vector<double> scores;

    FileNode params = args.params_file_node;
    int step = params["step"];
    float scale = params["scale"];
    int min_neighbs = params["min_neighbs"];
    int group_rect = params["group_rect"];

    Detector detector(classifier, Size(227, 227),
                      step, step, scale, min_neighbs, group_rect);

    for (size_t i = 0; i < args.filenames.size(); i++) {
        Mat img = imread(args.filenames[i], cv::IMREAD_COLOR);
        cout << "Processing " << args.filenames[i] << endl;

        detector.DetectMultiScale(img, labels, scores, rects);

        out << args.filenames[i] << endl << rects.size() << endl;
        for (size_t j = 0; j < rects.size(); j++) {
            out << rects[j].x << " " << rects[j].y << " "
                << rects[j].width << " " << rects[j].height << " "
                << scores[j] << " " << endl;
        }
        labels.clear();
        scores.clear();
        rects.clear();
    }
}

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

    ofstream out(args.input_path + "/result.txt");
    if (!out.is_open()) {
        cout << "Problems with creating output file\n";
        cout << help;
        return 1;
    }

    detect(classifier, args, out);

    out.close();
    return 0;
}
