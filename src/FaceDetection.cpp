#include "FaceDetection.h"

FaceDetection::FacialFeaturesDetector::FacialFeaturesDetector(std::string classifierDir, std::string asmModelsDir){
    std::string face_cascade_name = classifierDir+"haarcascade_frontalface_alt.xml";
    std::string eyes_cascade_name = classifierDir+"haarcascade_eye_tree_eyeglasses.xml";
    std::string mouth_cascade_name = classifierDir+"haarcascade_mcs_mouth.xml";
    std::string nose_cascade_name = classifierDir+"haarcascade_mcs_nose.xml";
    this->m_modelDir=asmModelsDir;
    this->m_asdModelPath=asmModelsDir+"color_asm75.model";
    if( !m_face_cascade.load( face_cascade_name ) ){ std::cout<<"--(!)Error loading Classifiers for Face"<<std::endl; throw("--(!)Error loading Classifiers");};
    if( !m_eyes_cascade.load( eyes_cascade_name ) ){ std::cout<<"--(!)Error loading Classifiers for Eyes"<<std::endl; throw("--(!)Error loading Classifiers");};
    if( !m_mouth_cascade.load( mouth_cascade_name ) ){ std::cout<<"--(!)Error loading Classifiers for Mouth"<<std::endl; throw("--(!)Error loading Classifiers");};
    if( !m_nose_cascade.load( nose_cascade_name ) ){ std::cout<<"--(!)Error loading Classifiers for Nose"<<std::endl; throw("--(!)Error loading Classifiers");};
    std::cout<<"Detector Initialized"<<std::endl;
}

void FaceDetection::FacialFeaturesDetector::loadDirectory(std::string pathToFolder){
    if(!boost::filesystem::exists(pathToFolder)){
        std::cout<<"The Image Dataset Folder doesn't exist."<<std::endl;
    }
    else{

        namespace fs = boost::filesystem;
        fs::path p(pathToFolder);
        fs::recursive_directory_iterator begin(p), end;
        std::cout<<"Loading Images..."<<std::endl;
        std::vector<fs::directory_entry> v(begin, end);
        std::cout << "There are " << v.size() << " files: \n";
        for(int i=0;i<v.size();i++){
            std::ostringstream oss;
            oss << v[i];
            std::string imagePath = v[i].path().string();
            std::cout<<imagePath<<std::endl;
            this->showCurrentImage(imagePath);
            this->processFileNames(imagePath);
        }
        for(int i=0;i<this->m_labels.size();i++){
            std::cout<<this->m_labels[i]<<std::endl;
        }

    }

}

void FaceDetection::FacialFeaturesDetector::showCurrentImage(std::string filePath){
    cv::Mat currentFrame = cv::imread(filePath,CV_LOAD_IMAGE_COLOR);
    this->readASMModel(m_asmModel,m_asdModelPath);
    this->searchAndFit(m_asmModel,m_face_cascade,currentFrame,0);
//    this->detectAndDisplay(currentFrame);
}

void FaceDetection::FacialFeaturesDetector::processFileNames(std::string filePath){
//    std::cout<<filePath<<std::endl;
    std::string processFileName = filePath;
    processFileName.erase(0,68);
    processFileName.erase(0,3);
    processFileName.erase(2,10);
//    std::cout<<processFileName<<std::endl;

    if(processFileName.compare("AN") ==0 ){
        this->m_labels.push_back(0);
    }

    else if(processFileName.compare("DI") == 0){
        this->m_labels.push_back(1);
    }

    else if(processFileName.compare("FE") == 0){
        this->m_labels.push_back(2);
    }

    else if(processFileName.compare("HA") == 0){
        this->m_labels.push_back(3);
    }


    else if(processFileName.compare("NE") == 0){
        this->m_labels.push_back(4);
    }


    else if(processFileName.compare("SA") == 0){
        this->m_labels.push_back(5);
    }


    else if(processFileName.compare("SU") == 0){
        this->m_labels.push_back(6);
    }

}

void FaceDetection::FacialFeaturesDetector::readASMModel(StatModel::ASMModel &asmModel, std::string modelPath){
    asmModel.loadFromFile(modelPath);
}

void FaceDetection::FacialFeaturesDetector::searchAndFit(StatModel::ASMModel &asmModel, cv::CascadeClassifier &objCascadeClassifer, cv::Mat &frame, int verboseL){
    /* Face Detection */
    std::vector<cv::Rect> faces;
    objCascadeClassifer.detectMultiScale(frame,faces,1.2,2,CV_HAAR_SCALE_IMAGE,cv::Size(60,60));
    std::vector<StatModel::ASMFitResult> fitResult = asmModel.fitAll(frame,faces,verboseL);
    this->m_pointResults=fitResult;
    StatModel::ASMModel* checking_asmModel;
    checking_asmModel=&asmModel;
    asmModel.showResult(frame,fitResult);
    cv::waitKey(0);
    this->pointAnalyzer(fitResult,checking_asmModel);

}

void FaceDetection::FacialFeaturesDetector::pointAnalyzer(std::vector<StatModel::ASMFitResult> &fitResults, StatModel::ASMModel *asmModel){
    std::vector<cv::Point > checkingPoints;
    fitResults[0].toPointList(checkingPoints);
    std::cout<<checkingPoints.size()<<std::endl;
    std::cout<<fitResults.size()<<std::endl;
    for(int i=0;i<checkingPoints.size();i++){
        std::cout<<"("<<checkingPoints[i].x<<","<<checkingPoints[i].y<<")"<<std::endl;
    }
}

void FaceDetection::FacialFeaturesDetector::detectAndDisplay(cv::Mat frame){
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    cv::RotatedRect box;
    cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    this->m_window_name="Facial Features";

    //-- Detect faces
    m_face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    for( std::size_t i = 0; i < faces.size(); i++ )
     {
       cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
       cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 2, 8, 0 );

       cv::Mat faceROI = frame_gray( faces[i] );
       std::vector<cv::Rect> eyes;

       //-- In each face, detect eyes
       m_eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

       for( std::size_t j = 0; j < eyes.size(); j++ )
        {
          cv::Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
          int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
          cv::circle( frame, eye_center, radius, cv::Scalar( 255, 0, 0 ), 3, 8, 0 );
        }

       std::vector<cv::Rect> mouth;
       std::vector<cv::Rect> nose;
       m_mouth_cascade.detectMultiScale( faceROI, mouth, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
       m_nose_cascade.detectMultiScale( faceROI, nose, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

       for(int k=0; k<mouth.size(); k++)
       {
           //Point center(faces[i].x+eyes[j].x+eyes[j].width*0.5, faces[i].y+eyes[j].y+eyes[j].height*0.5);
           cv::Point center_ell( faces[i].x + mouth[k].x + mouth[k].width*0.5, faces[i].y+(faces[i].height)*2/3+ mouth[k].y + mouth[k].height*0.5 );
           cv::Size RectSize(mouth[k].width,mouth[k].height);
           int axes = cvRound((mouth[k].width+mouth[k].height)*0.20);

           box = cv::RotatedRect(center_ell,RectSize,0);
           cv::ellipse(frame, box, cv::Scalar(255,0,0),2,8);


           //circle(cap_img, center, radius, Scalar(255,0,0), 2, 8, 0);
       }

       for( std::size_t j = 0; j < nose.size(); j++ )
        {
          cv::Point nose_center( faces[i].x + nose[j].x + nose[j].width/2, faces[i].y + nose[j].y + nose[j].height/2 );
          int radius = cvRound( (nose[j].width + nose[j].height)*0.25 );
          cv::circle( frame, nose_center, radius, cv::Scalar( 0, 255, 0 ), 3, 8, 0 );
        }


     }
    //-- Show what you got
    cv::namedWindow(this->m_window_name,CV_WINDOW_AUTOSIZE);
    cv::imshow( this->m_window_name,frame);
    cv::waitKey(0);

}
