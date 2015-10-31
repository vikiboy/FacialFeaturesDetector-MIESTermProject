/* Author : Vikram Mohanty
 * IIT Kharagpur
 * vikram.mohanty.1993@gmail.com
 * */

#include "FaceDetection.h"

FaceDetection::FacialFeaturesDetector::FacialFeaturesDetector(std::string classifierDir, std::string asmModelsDir){
    std::string face_cascade_name = classifierDir+"haarcascade_frontalface_alt2.xml";
    std::string eyes_cascade_name = classifierDir+"haarcascade_lefteye_2splits.xml";
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
            this->m_Matrix.push_back(this->m_perImage);
            this->m_perImage.clear();
            std::cout<<"Size of the m_Matrix : "<<m_Matrix.size()<<std::endl;
            cv::waitKey(0);
//            std::cout<<"Size of the perImage vector : "<<(this->m_perImage.size())/3<<std::endl;
        }
        this->createResultsFile();
        this->processResults(m_Matrix,m_labels);

//        for(int i=0;i<this->m_labels.size();i++){
//            std::cout<<this->m_labels[i]<<std::endl;
//        }

    }

}

void FaceDetection::FacialFeaturesDetector::showCurrentImage(std::string filePath){
    cv::Mat currentFrame = cv::imread(filePath,CV_LOAD_IMAGE_COLOR);
    this->readASMModel(m_asmModel,m_asdModelPath);
    this->searchAndFit(m_asmModel,m_face_cascade,currentFrame,0);
    this->detectAndDisplay(currentFrame);
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
        std::cout<<"Expression : 0"<<std::endl;
    }

    else if(processFileName.compare("DI") == 0){
        this->m_labels.push_back(1);
        std::cout<<"Expression : 1"<<std::endl;
    }

    else if(processFileName.compare("FE") == 0){
        this->m_labels.push_back(2);
        std::cout<<"Expression : 2"<<std::endl;
    }

    else if(processFileName.compare("HA") == 0){
        this->m_labels.push_back(3);
        std::cout<<"Expression : 3"<<std::endl;
    }


    else if(processFileName.compare("NE") == 0){
        this->m_labels.push_back(4);
        std::cout<<"Expression : 4"<<std::endl;
    }


    else if(processFileName.compare("SA") == 0){
        this->m_labels.push_back(5);
        std::cout<<"Expression : 5"<<std::endl;
    }


    else if(processFileName.compare("SU") == 0){
        this->m_labels.push_back(6);
        std::cout<<"6"<<std::endl;
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
    this->pointAnalyzer(fitResult,checking_asmModel);
//    cv::waitKey(0);

}

void FaceDetection::FacialFeaturesDetector::pointAnalyzer(std::vector<StatModel::ASMFitResult> &fitResults, StatModel::ASMModel *asmModel){
    std::vector<cv::Point > checkingPoints;
    fitResults[0].toPointList(checkingPoints);

    // Analyzing the chin
    std::vector<double> chinSum;
    for(int i=8;i<=21;i++){

        chinSum.push_back(cv::norm(cv::Mat(checkingPoints[i]),cv::Mat(checkingPoints[i+1])));
    }
    double chinFinalSum=0;
    for(int i=0;i<chinSum.size();i++){
        chinFinalSum+=chinSum[i];
    }
    std::cout<<"Chin Circumference: "<<chinFinalSum<<std::endl;
    double chinExtreme_distance = cv::norm(cv::Mat((checkingPoints[19]+checkingPoints[20])*.5),cv::Mat((checkingPoints[9]+checkingPoints[10])*.5));
    std::cout<<"Chin Extreme Distance: "<<chinExtreme_distance<<std::endl;

    std::vector<double> chinAttribs; //Final Chin Attribs Stored here
    chinAttribs.push_back(chinFinalSum); //Adding the Chin Circumference
    chinAttribs.push_back(chinExtreme_distance); //Adding the distance between between Chin extremities

    //Analyzing the Eye and Eyebrow Distance
    std::vector<double> EyesEyebrowsAttribs;

    //Left Eyebrow
    cv::Mat leftEyebrow_centroid(cv::Point(0,0));
    for(int i=28;i<=33;i++){//Left Eyebrow Centroid
        leftEyebrow_centroid+=cv::Mat(checkingPoints[i]);
//        std::cout<<"Left Eye "<<i-27<<" : "<<checkingPoints[i]<<std::endl;
    }
    leftEyebrow_centroid=leftEyebrow_centroid*(1.0/6.0);
//    std::cout<<"Left Eyebrow Centroid : "<<leftEyebrow_centroid<<std::endl;

    //Right Eyebrow
    cv::Mat rightEyebrow_centroid(cv::Point(0,0));
    for(int i=22;i<=27;i++){//Right Eyebrow Centroid
        rightEyebrow_centroid+=cv::Mat(checkingPoints[i]);
//        std::cout<<"Right Eye "<<i-21<<" : "<<checkingPoints[i]<<std::endl;
    }
    rightEyebrow_centroid=rightEyebrow_centroid*(1.0/6.0);
//    std::cout<<"Right Eyebrow Centroid : "<<leftEyebrow_centroid<<std::endl;

    //Distance between left and right eyebrow extremes - near,far and the centroid
    double eyebrow_nearExtreme=cv::norm(cv::Mat(checkingPoints[25]),cv::Mat(checkingPoints[31]));  //attribs
    double eyebrow_farExtreme=cv::norm(cv::Mat(checkingPoints[22]),cv::Mat(checkingPoints[28])); //attribs
    double eyebrow_centroidDistance = cv::norm(leftEyebrow_centroid,rightEyebrow_centroid); //attribs


    std::cout<<"Eyebrow Near Extreme : "<<eyebrow_nearExtreme<<" , Eyebrow Far Extreme : "<<eyebrow_farExtreme<<" , Eyebrow Centroid : "<<eyebrow_centroidDistance<<std::endl;

    //Left Eye
    cv::Mat leftEye_centroid(cv::Point(0,0));
    cv::Mat rightEye_centroid(cv::Point(0,0));
    for(int i=34;i<=38;i++){
        leftEye_centroid+=cv::Mat(checkingPoints[i]);
    }
    leftEye_centroid=leftEye_centroid*(1.0/5.0);
    //Right Eye
    for(int i=39;i<=43;i++){
        rightEye_centroid+=cv::Mat(checkingPoints[i]);
    }
    rightEye_centroid=rightEye_centroid*(1.0/5.0);
//    std::cout<<"Left Eye Centroid : "<<leftEye_centroid<<" Right Eye Centroid : "<<rightEye_centroid<<std::endl;

    //Distance between top and bottom
    double avg_topbottomEyeDistance=(cv::norm(cv::Mat(checkingPoints[35]),cv::Mat(checkingPoints[37]))+cv::norm(cv::Mat(checkingPoints[40]),cv::Mat(checkingPoints[42])))/2; //attribs
    double avg_leftrightEyeDistance=(cv::norm(cv::Mat(checkingPoints[34]),cv::Mat(checkingPoints[36]))+cv::norm(cv::Mat(checkingPoints[39]),cv::Mat(checkingPoints[41])))/2; //attribs
    double centroidDistanceLeftEyebrowEye=cv::norm(leftEyebrow_centroid,leftEye_centroid); //attribs
    double centroidDistanceRightEyebrowEye=cv::norm(rightEyebrow_centroid,rightEye_centroid); //attribs

    EyesEyebrowsAttribs.push_back(eyebrow_centroidDistance); //Adding the Distance between the centroids of the two eyebrows
    EyesEyebrowsAttribs.push_back(eyebrow_farExtreme); //Adding the Distance between the far extremities of the two eyebrows
    EyesEyebrowsAttribs.push_back(eyebrow_nearExtreme); //Adding the Distance between the near extremities of the two eyebrows
    EyesEyebrowsAttribs.push_back(avg_topbottomEyeDistance); //Adding the average distance between the top and bottom of both eyes
    EyesEyebrowsAttribs.push_back(avg_leftrightEyeDistance); //Adding the average distance between the left and right extremities of both eyes
    EyesEyebrowsAttribs.push_back(centroidDistanceLeftEyebrowEye); //Adding the distance between left eyebrow and left eye
    EyesEyebrowsAttribs.push_back(centroidDistanceRightEyebrowEye); //Adding the distance between right eyebrow and right eye

    std::cout<<"Top Bottom Eye Distance : "<<avg_topbottomEyeDistance<<" , Left Right Eye Distance : "<<avg_leftrightEyeDistance<<std::endl;
    std::cout<<"Left Eye Eyebrow Distance : "<<centroidDistanceLeftEyebrowEye<<" , Right Eye Eyebrow Distance : "<<centroidDistanceRightEyebrowEye<<std::endl;

    //Analyzing the Mouth, Nose and Chin Features

    std::vector<double> mouthNoseAttribs;

    double avg_noseExtremeDistance=(cv::norm(cv::Mat(checkingPoints[46]),cv::Mat(checkingPoints[50]))+cv::norm(cv::Mat(checkingPoints[47]),cv::Mat(checkingPoints[49])))/2; //attribs
    cv::Mat outerMouth_centroid(cv::Point(0,0));
    for(int i=55;i<=66;i++){
        outerMouth_centroid+=cv::Mat(checkingPoints[i]);
    }
    outerMouth_centroid=outerMouth_centroid/12.0;
    double mouthNoseDistance =cv::norm(outerMouth_centroid,cv::Mat(checkingPoints[74])); //attribs
    double mouth_sideExtremeDistance=cv::norm(cv::Mat((checkingPoints[55]+checkingPoints[66])*.5),cv::Mat((checkingPoints[61]+checkingPoints[62])*.5)); //attribs
    double mouth_topExtremeDistance=cv::norm(cv::Mat((checkingPoints[57]+checkingPoints[58]+checkingPoints[59]*(1/3.0))),cv::Mat((checkingPoints[63]+checkingPoints[64]+checkingPoints[65])*(1/3.0))); //attribs
    double noseChinDistance=(cv::norm(cv::Mat((checkingPoints[45]+checkingPoints[46])*.5),cv::Mat((checkingPoints[19]+checkingPoints[20])*.5))
            +cv::norm(cv::Mat((checkingPoints[50]+checkingPoints[51])*.5),cv::Mat((checkingPoints[19]+checkingPoints[20])*.5))/2); //attribs

    std::cout<<"Mouth to Nose Distance : "<<mouthNoseDistance<<" , Mouth Side Extremes : "<<mouth_sideExtremeDistance<<" , Mouth Top Bottom Distance : "<<mouth_topExtremeDistance<<std::endl;
    std::cout<<"Nose Extreme Distance : "<<avg_noseExtremeDistance<<std::endl;

    mouthNoseAttribs.push_back(mouthNoseDistance); //Adding the Distance between the Mouth to Nose
    mouthNoseAttribs.push_back(mouth_sideExtremeDistance); //Adding the Distance between side extremities of the mouth
    mouthNoseAttribs.push_back(mouth_topExtremeDistance); //Adding the Distance between the top extremities of the mouth
    mouthNoseAttribs.push_back(noseChinDistance); //Adding the Distance between the chin and nose

    this->m_perImage.push_back(chinAttribs);
    this->m_perImage.push_back(EyesEyebrowsAttribs);
    this->m_perImage.push_back(mouthNoseAttribs);

}

void FaceDetection::FacialFeaturesDetector::detectAndDisplay(cv::Mat frame){
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    cv::RotatedRect box;
    cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    this->m_window_name="Facial Features";

    std::vector<double> cascadeAttribs;

    //-- Detect faces
    m_face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( std::size_t i = 0; i < faces.size(); i++ )
     {
       cv::Mat eyeimg;
       cv::Rect roi1 = cv::Rect((int)faces[i].x,                   /* x = start from leftmost */
                    (int)faces[i].y,                          /* y = from the leftmost */
                    (int) faces[i].width,                       /* width = same width with the face */
                    (int)(faces[i].height)/2 ) ;               /* height = 1/2 of face height */
       eyeimg = frame_gray(roi1);
       cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
       cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 2, 8, 0 );

       cv::Mat faceROI = frame_gray( faces[i] );
       std::vector<cv::Rect> eyes;

       //-- In each face, detect eyes
       m_eyes_cascade.detectMultiScale( eyeimg, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

       for( std::size_t j = 0; j < eyes.size(); j++ )
        {
          cv::Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
          int radius = cvRound( (eyes[j].width + eyes[j].height)*0.20 );
          cv::circle( frame, eye_center, radius, cv::Scalar( 255, 0, 0 ), 2, 8, 0 );
          if(eyes.size()==2){
              cascadeAttribs.push_back(radius);//attribs
              cascadeAttribs.push_back(eyes[j].width);//attribs
              cascadeAttribs.push_back(eyes[j].height);//attribs
          }
          else if(eyes.size()==1){
              cascadeAttribs.push_back(radius);//attribs
              cascadeAttribs.push_back(eyes[j].width);//attribs
              cascadeAttribs.push_back(eyes[j].height);//attribs
              cascadeAttribs.push_back(radius);//attribs
              cascadeAttribs.push_back(eyes[j].width);//attribs
              cascadeAttribs.push_back(eyes[j].height);//attribs
          }
        }

       cv::Mat mouthImg;
       cv::Rect roi2 = cv::Rect((int)faces[i].x,                   /* x = start from leftmost */
                (int)faces[i].y+((faces[i].height)*2/3),    /* y = a few pixels from the top */
               (int) faces[i].width,                       /* width = same width with the face */
                (int)(faces[i].height)/3 ) ;               /* height = 1/3 of face height */
       mouthImg = frame_gray(roi2);       std::vector<cv::Rect> mouth;
       std::vector<cv::Rect> nose;
       m_mouth_cascade.detectMultiScale(mouthImg, mouth, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
       m_nose_cascade.detectMultiScale( faceROI, nose, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

       for(int k=0; k<mouth.size(); k++)
       {
           //Point center(faces[i].x+eyes[j].x+eyes[j].width*0.5, faces[i].y+eyes[j].y+eyes[j].height*0.5);
           cv::Point center_ell( faces[i].x + mouth[k].x + mouth[k].width*0.5, faces[i].y+(faces[i].height)*2/3+ mouth[k].y + mouth[k].height*0.5 );
           cv::Size RectSize(mouth[k].width,mouth[k].height);
           int axes = cvRound((mouth[k].width+mouth[k].height)*0.20);
//           std::cout<<"Mouth Width : "<<mouth[k].width;
//           std::cout<<" , Mouth Height : "<<mouth[k].height<<std::endl;

           box = cv::RotatedRect(center_ell,RectSize,0);
           cv::ellipse(frame, box, cv::Scalar(255,0,0),2,8);
           cascadeAttribs.push_back(mouth[k].width); //attribs
           cascadeAttribs.push_back(mouth[k].height); //attribs
           //circle(cap_img, center, radius, Scalar(255,0,0), 2, 8, 0);
       }

//       for( std::size_t j = 0; j < nose.size(); j++ )
//        {
//          cv::Point nose_center( faces[i].x + nose[j].x + nose[j].width/2, faces[i].y + nose[j].y + nose[j].height/2 );
//          int radius = cvRound( (nose[j].width + nose[j].height)*0.25 );
//          cv::circle( frame, nose_center, radius, cv::Scalar( 0, 255, 0 ), 3, 8, 0 );
//        }


     }
    this->m_perImage.push_back(cascadeAttribs);
    //-- Show what you got
    cv::namedWindow(this->m_window_name,CV_WINDOW_AUTOSIZE);
    cv::imshow( this->m_window_name,frame);
    cv::waitKey(0);

}

void FaceDetection::FacialFeaturesDetector::processResults(std::vector<std::vector<std::vector<double> > > attribs, std::vector<double> outputLabels){
    double chinFinalSum,chinExtremeDistance,eyebrowcentroidDistance,eyebrowfarExtreme,eyebrownearExtreme,avgtopBottomEyeDistance,
            avgleftRightEyeDistance,LeftEyebrow_Eye,RightEyebrow_Eye,mouthNose_Distance,mouthsideExtremeDistance,mouthtopExtremeDistance,
            noseChinDistance,leftEyeRadius,leftEyeWidth,leftEyeHeight,rightEyeRadius,rightEyeWidth,rightEyeHeight,mouthWidth,mouthHeight;

    for(int i=0;i<attribs.size();i++){
        for(int j=0;j<attribs[i].size();j++){
            for(int k=0;k<attribs[i][j].size();k++){
                if(j==0){
                    if(k==0){
                        chinFinalSum=attribs[i][j][k];
                    }
                    else if(k==1){
                        chinExtremeDistance=attribs[i][j][k];
                    }
                }
                else if(j==1){
                    if(k==0){
                        eyebrowcentroidDistance=attribs[i][j][k];
                    }
                    else if(k==1){
                        eyebrowfarExtreme=attribs[i][j][k];
                    }
                    else if(k==2){
                        eyebrownearExtreme=attribs[i][j][k];
                    }
                    else if(k==3){
                        avgtopBottomEyeDistance=attribs[i][j][k];
                   }
                    else if(k==4){
                        avgleftRightEyeDistance=attribs[i][j][k];
                    }
                    else if(k==5){
                        LeftEyebrow_Eye=attribs[i][j][k];
                    }
                    else if(k==6){
                        RightEyebrow_Eye=attribs[i][j][k];
                    }

                }
                else if(j==2){
                    if(k==0){
                        mouthNose_Distance=attribs[i][j][k];
                    }
                    else if(k==1){
                        mouthsideExtremeDistance=attribs[i][j][k];
                    }
                    else if(k==2){
                        mouthtopExtremeDistance=attribs[i][j][k];
                    }
                    else if(k==3){
                        noseChinDistance=attribs[i][j][k];
                    }
                }
                else if(j==3){
                    if(k==0){
                        leftEyeRadius=attribs[i][j][k];
                    }
                    else if(k==1){
                        leftEyeWidth=attribs[i][j][k];
                    }
                    else if(k==2){
                        leftEyeHeight=attribs[i][j][k];
                    }
                    else if(k==3){
                        rightEyeRadius=attribs[i][j][k];
                    }
                    else if(k==4){
                        rightEyeWidth=attribs[i][j][k];
                    }
                    else if(k==5){
                        rightEyeHeight=attribs[i][j][k];
                    }
                    else if(k==6){
                        mouthWidth=attribs[i][j][k];
                    }
                    else if(k==7){
                        mouthHeight=attribs[i][j][k];
                    }
                }
            }
        }
        this->saveResults(chinFinalSum,chinExtremeDistance,eyebrowcentroidDistance,eyebrowfarExtreme,eyebrownearExtreme,avgtopBottomEyeDistance,avgleftRightEyeDistance,LeftEyebrow_Eye,
                          RightEyebrow_Eye,mouthNose_Distance,mouthsideExtremeDistance,mouthtopExtremeDistance,noseChinDistance,leftEyeRadius,leftEyeWidth,leftEyeHeight,rightEyeRadius,
                          rightEyeWidth,rightEyeHeight,mouthWidth,mouthHeight,outputLabels[i]);
    }

}

void FaceDetection::FacialFeaturesDetector::saveResults(double chinCircumference, double chinExtreme_distance, double eyebrow_centroidDistance, double eyebrow_farExtreme, double eyebrow_nearExtreme, double avg_topbottomEyeDistance,
                                                        double avg_leftrightEyeDistance, double LeftEyebrowEye, double RightEyebrowEye, double mouthNoseDistance, double mouth_sideExtremeDistance, double mouth_topExtremeDistance,
                                                        double nose_ChinDistance, double leftEyeRadius,double leftEyeWidth,double leftEyeHeight,double rightEyeRadius, double rightEyeWidth, double rightEyeHeight,
                                                        double mouthWidth, double mouthHeight, double outputLabel){
    std::ofstream outfile;
    outfile.open(m_resultsFile.c_str(),std::ofstream::app);
    outfile<<chinCircumference<<"\t"<<chinExtreme_distance<<"\t"<<eyebrow_centroidDistance<<"\t"<<eyebrow_farExtreme<<"\t"<<eyebrow_nearExtreme<<"\t"<<avg_topbottomEyeDistance<<"\t"
             <<avg_leftrightEyeDistance<<"\t"<<LeftEyebrowEye<<"\t"<<RightEyebrowEye<<"\t"<<mouthNoseDistance<<"\t"<<mouth_sideExtremeDistance<<"\t"<<mouth_topExtremeDistance<<"\t"<<nose_ChinDistance<<"\t"
            <<leftEyeRadius<<"\t"<<leftEyeWidth<<"\t"<<leftEyeHeight<<"\t"<<rightEyeRadius<<"\t"<<rightEyeWidth<<"\t"<<rightEyeHeight<<"\t"
           <<mouthWidth<<"\t"<<mouthHeight<<"\t"<<outputLabel<<"\n";
    outfile.close();
}

void FaceDetection::FacialFeaturesDetector::createResultsFile(){
    m_resultsFile="FacialFeaturesJaffe.csv";
    std::ofstream myfile;
    myfile.open(m_resultsFile.c_str(),std::ofstream::app);
    myfile<<"Chin_Circumference"<<"\t"<<"Chin_extremitiesDistance"<<"\t"<<"EyeBrow_CentroidDistance"<<"\t"<<"EyeBrow_farExtreme"<<"\t"<<"EyeBrow_nearExtreme"<<"\t"<<"avg_TopBottomEyeDistance"<<"\t"
             <<"avg_LeftRightEyeDistance"<<"\t"<<"LeftEyebrowEyes"<<"\t"<<"RightEyebrowEyes"<<"\t"<<"mouthNoseDistance"<<"\t"<<"mouthSideExtreme"<<"\t"<<"mouthTopExtremeDistance"<<"\t"<<"noseChinDistance"<<"\t"
            <<"EyeRadius1"<<"\t"<<"EyeWidth1"<<"\t"<<"EyeHeight1"<<"\t"<<"EyeRadius2"<<"\t"<<"EyeWidth2"<<"\t"<<"EyeHeight2"<<"\t"<<"MouthWidth"<<"\t"<<"MouthHeight"<<"\t"
           <<"Emotion"<<"\n";
    myfile.close();

}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == CV_EVENT_LBUTTONDOWN )
     {
          std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if  ( event == CV_EVENT_RBUTTONDOWN )
     {
          std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if  ( event == CV_EVENT_MBUTTONDOWN )
     {
          std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
}
