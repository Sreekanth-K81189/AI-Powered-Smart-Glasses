#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "OnnxDetector.hpp"

using ms  = std::chrono::milliseconds;
using clk = std::chrono::steady_clock;

static const std::vector<std::string> COCO_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

static std::string classLabel(int id){
    if(id>=0&&id<(int)COCO_NAMES.size()) return COCO_NAMES[id];
    return "id="+std::to_string(id);
}

static bool tryUSB(cv::VideoCapture& cap,int idx,int backend,const char* name){
    cap.release();
    std::cout<<"[VERIFY] Trying index="<<idx<<" backend="<<name<<"\n";
    backend>=0 ? cap.open(idx,backend) : cap.open(idx);
    if(!cap.isOpened()) return false;
    cap.set(cv::CAP_PROP_BUFFERSIZE,1);
    cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
    cap.set(cv::CAP_PROP_FPS,30);
    for(int i=0;i<50;++i){
        cv::Mat f; if(cap.read(f)&&!f.empty()){
            std::cout<<"[VERIFY]   OK ("<<f.cols<<"x"<<f.rows<<")\n"; return true;
        }
        std::this_thread::sleep_for(ms(100));
    }
    cap.release(); return false;
}

int main(){
#ifdef _WIN32
    AllocConsole(); FILE* fd=nullptr;
    freopen_s(&fd,"CONOUT$","w",stdout);
    freopen_s(&fd,"CONOUT$","w",stderr);
#endif
    const std::string mdl="models/yolo/yolov8x_fp16.onnx";
    std::cout<<"[VERIFY] Loading: "<<mdl<<"\n";
    OnnxDetector detector(mdl);

    cv::namedWindow("object_detection_verify",cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap; bool ok=false;
    for(int i=0;i<=5&&!ok;++i) ok=tryUSB(cap,i,cv::CAP_DSHOW,"DSHOW");
    for(int i=0;i<=5&&!ok;++i) ok=tryUSB(cap,i,cv::CAP_MSMF,"MSMF");
    for(int i=0;i<=3&&!ok;++i) ok=tryUSB(cap,i,-1,"AUTO");
    if(!ok){ std::cerr<<"No camera.\n"; return 1; }

    int fpsCount=0; float fps=0.f;
    auto fpsLast=clk::now();
    constexpr int INFER_MS=80;
    auto inferLast=clk::now()-ms(INFER_MS*2);
    std::vector<Detection> lastDets;

    for(;;){
        cv::Mat frame;
        if(!cap.read(frame)||frame.empty()) break;
        auto now=clk::now();
        if((int)std::chrono::duration_cast<ms>(now-inferLast).count()>=INFER_MS){
            lastDets=detector.detect(frame);
            inferLast=clk::now();
        }
        for(const auto& d:lastDets){
            float conf=std::max(0.f,std::min(1.f,d.confidence));
            std::string lbl=classLabel(d.classId)+" "+cv::format("%.0f%%",conf*100.f);
            cv::rectangle(frame,d.bbox,cv::Scalar(0,255,255),2);
            int bl=0;
            cv::Size sz=cv::getTextSize(lbl,cv::FONT_HERSHEY_SIMPLEX,0.55,1,&bl);
            int ty=std::max(d.bbox.y-4,sz.height+4);
            cv::rectangle(frame,cv::Point(d.bbox.x,ty-sz.height-4),
                          cv::Point(d.bbox.x+sz.width+4,ty+2),
                          cv::Scalar(0,255,255),cv::FILLED);
            cv::putText(frame,lbl,cv::Point(d.bbox.x+2,ty-2),
                        cv::FONT_HERSHEY_SIMPLEX,0.55,cv::Scalar(0,0,0),1);
        }
        ++fpsCount;
        float sec=std::chrono::duration<float>(clk::now()-fpsLast).count();
        if(sec>=1.f){ fps=fpsCount/sec; fpsCount=0; fpsLast=clk::now(); }
        std::string hud=cv::format("FPS:%.1f  ORT:%s  %dx%d",
            fps,detector.isUsingCuda()?"CUDA":"CPU",frame.cols,frame.rows);
        cv::putText(frame,hud,cv::Point(10,28),
                    cv::FONT_HERSHEY_SIMPLEX,0.75,cv::Scalar(0,255,0),2);
        cv::imshow("object_detection_verify",frame);
        if(cv::waitKey(1)==27) break;
    }
    return 0;
}
