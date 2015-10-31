/* Author : Vikram Mohanty
 * IIT Kharagpur
 * vikram.mohanty.1993@gmail.com
 * */


#ifndef __FACEDETECTION__H__
#define __FACEDETECTION__H__


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "asmmodel.h"
#include "modelfile.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/lambda/bind.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/ptr_container/ptr_vector.hpp>


namespace FaceDetection {
    class FacialFeaturesDetector
    {
    public:
        FacialFeaturesDetector(std::string classifierDir,std::string asmModelsDir);

        //! Loading the training data
        void loadDirectory(std::string pathToFolder);

        //! For detecting with HAAR Classifiers - Face, Eyes, Mouth and Nose
        void detectAndDisplay( cv::Mat frame );

        //! Load the current image
        void showCurrentImage(std::string filePath);

        //! Process filenames
        void processFileNames(std::string filePath);

        //! Load an ASM Model
        void readASMModel(StatModel::ASMModel & asmModel, std::string modelPath);

        //! Fit an ASM Model
        void searchAndFit(StatModel::ASMModel &asmModel,cv::CascadeClassifier &objCascadeClassifer,cv::Mat &frame,int verboseL=0);

        //! Analyze the vector of feature points
        void pointAnalyzer(std::vector<StatModel::ASMFitResult> &fitResults,StatModel::ASMModel* asmModel);

        //! Process results
        void processResults(std::vector<std::vector<std::vector<double> > > attribs,std::vector<double> outputLabels);

        //! Save results to file
        void saveResults(double chinCircumference, double chinExtreme_distance, double eyebrow_centroidDistance, double eyebrow_farExtreme, double eyebrow_nearExtreme,
                                                                double avg_topbottomEyeDistance, double avg_leftrightEyeDistance, double LeftEyebrowEye, double RightEyebrowEye, double mouthNoseDistance,
                                                                double mouth_sideExtremeDistance, double mouth_topExtremeDistance, double nose_ChinDistance, double leftEyeRadius,double leftEyeWidth,
                                                                double leftEyeHeight,double rightEyeRadius, double rightEyeWidth, double rightEyeHeight, double mouthWidth, double mouthHeight, double outputLabel);

        //! Create a results file
        void createResultsFile();

    private:

        std::vector<std::vector<std::vector<double> > > m_Matrix; //to store the attributes along with the output label
        std::vector<std::vector<double> > m_perImage;
        std::vector<double> m_labels; //output label
        std::vector<StatModel::ASMFitResult> m_pointResults; //for analyzing points
        StatModel::ASMModel m_asmModel;
        std::string m_window_name;
        cv::CascadeClassifier m_eyes_cascade;
        cv::CascadeClassifier m_face_cascade;
        cv::CascadeClassifier m_mouth_cascade;
        cv::CascadeClassifier m_nose_cascade;
        std::string m_asdModelPath;
        std::string m_modelDir;
        std::string m_fileName;

        std::string m_resultsFile;
    };
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata);

#endif
