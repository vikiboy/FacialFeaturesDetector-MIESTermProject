/* Author : Vikram Mohanty
 * IIT Kharagpur
 * vikram.mohanty.1993@gmail.com
 * */

#include "FaceDetection.h"

cv::RNG rng(12345);
std::string classifierDir = "../FaceDetection/data/haarCascades/";
std::string asmModelsDir = "../FaceDetection/data/asmModels/";

int main( int argc, char* argv[] )
{
  FaceDetection::FacialFeaturesDetector detector(classifierDir,asmModelsDir);
  detector.loadDirectory("/home/vikiboy/Desktop/FaceDetection-MIES/datasets/jaffeimages/jaffe");
//  detector.captureImage();
  return 0;

}


