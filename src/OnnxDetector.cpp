#include <algorithm>
#include <iostream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "OnnxDetector.hpp"

static const char* COCO_NAMES[80] = {
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
const char* OnnxDetector::cocoName(int id){
    if(id>=0&&id<80) return COCO_NAMES[id];
    return "unknown";
}

static bool strhas(const std::string& s, const char* sub){
    std::string l=s; for(auto& c:l) c=(char)tolower(c);
    return l.find(sub)!=std::string::npos;
}

static std::string joinProviders(const std::vector<std::string>& providers) {
    std::ostringstream oss;
    for (size_t i = 0; i < providers.size(); ++i) {
        oss << providers[i];
        if (i + 1 < providers.size()) oss << ", ";
    }
    return oss.str();
}

// Ã¢â€â‚¬Ã¢â€â‚¬ Constructor Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
OnnxDetector::OnnxDetector(const std::string& modelPath)
    : env_(ORT_LOGGING_LEVEL_WARNING,"OnnxDetector")
{
    Ort::SessionOptions opts;
    opts.SetLogSeverityLevel(4);
    opts.SetExecutionMode(ORT_SEQUENTIAL);
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    try {
        char** provs = nullptr;
        int provCount = 0;
        if (Ort::GetApi().GetAvailableProviders(&provs, &provCount) == nullptr) {
            std::vector<std::string> names;
            names.reserve((size_t)std::max(0, provCount));
            for (int i = 0; i < provCount; ++i) names.emplace_back(provs[i] ? provs[i] : "");
            Ort::GetApi().ReleaseAvailableProviders(provs, provCount);
            std::cerr << "[OnnxDetector] Available EPs: " << joinProviders(names) << "\n";
        }
    } catch (...) {
        // ignore provider listing errors (shouldn't happen)
    }

    try {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        // Some onnxruntime versions do not expose cuda.enable_cuda_graph.
        // Rely on default options for CUDA graph configuration.
        opts.AppendExecutionProvider_CUDA(cuda);
        usingCuda_ = true;
        std::cerr << "[OnnxDetector] Using CUDA execution provider.\n";
    } catch (const Ort::Exception& e) {
        usingCuda_ = false;
        std::cerr << "[OnnxDetector] CUDA EP not available, using CPU. ORT error: "
                  << e.what() << "\n";
    } catch (const std::exception& e) {
        usingCuda_ = false;
        std::cerr << "[OnnxDetector] CUDA EP not available, using CPU. std::exception: "
                  << e.what() << "\n";
    } catch (...) {
        usingCuda_ = false;
        std::cerr << "[OnnxDetector] CUDA EP not available, using CPU. (unknown error)\n";
    }
#ifdef _WIN32
    std::wstring wp(modelPath.begin(),modelPath.end());
    session_=Ort::Session(env_,wp.c_str(),opts);
#else
    session_=Ort::Session(env_,modelPath.c_str(),opts);
#endif

    Ort::AllocatorWithDefaultOptions alloc;
    numOutputs_=session_.GetOutputCount();
    for(size_t i=0;i<numOutputs_;++i){
        auto n=session_.GetOutputNameAllocated(i,alloc);
        outputNames_.push_back(n.get());
    }
    { auto n=session_.GetInputNameAllocated(0,alloc); inputName_=n.get(); }

    // Print output shapes
    std::cerr<<"[OnnxDetector] Model outputs ("<<numOutputs_<<"):\n";
    for(size_t i=0;i<numOutputs_;++i){
        auto info=session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto sh=info.GetShape();
        std::cerr<<"  ["<<i<<"] \""<<outputNames_[i]<<"\" [";
        for(size_t j=0;j<sh.size();++j) std::cerr<<sh[j]<<(j+1<sh.size()?",":"");
        std::cerr<<"]\n";
    }

    // Ã¢â€â‚¬Ã¢â€â‚¬ Detect output format Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    if(numOutputs_==1){
        auto sh=session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // [1, N, 6]  ->  NMS6 (x1,y1,x2,y2,conf,cls)
        if(sh.size()==3 && sh[2]==6){
            outputFormat_=OutputFormat::NMS6;
            maxDet_=(int)sh[1];
            std::cerr<<"[OnnxDetector] Format: NMS6 [1,"<<maxDet_<<",6]\n";
        }
        // [1, 84, N] ->  RAW
        else if(sh.size()==3 && sh[1]>=84){
            outputFormat_=OutputFormat::RAW;
            std::cerr<<"[OnnxDetector] Format: RAW [1,"<<sh[1]<<","<<sh[2]<<"]\n";
        }
        else {
            outputFormat_=OutputFormat::RAW;
            std::cerr<<"[OnnxDetector] Format: RAW (unknown single-output shape)\n";
        }
    }
    else if(numOutputs_>=4){
        outputFormat_=OutputFormat::NMS4;
        std::cerr<<"[OnnxDetector] Format: NMS4\n";
        // Resolve tensor roles by name
        idxNumDets=-1; idxBoxes=-1; idxScores=-1; idxClasses=-1;
        for(int i=0;i<(int)numOutputs_;++i){
            const std::string& nm=outputNames_[i];
            if     (strhas(nm,"num"))                          idxNumDets=i;
            else if(strhas(nm,"box"))                          idxBoxes  =i;
            else if(strhas(nm,"score")||strhas(nm,"conf"))     idxScores =i;
            else if(strhas(nm,"class")||strhas(nm,"label"))    idxClasses=i;
        }
        if(idxNumDets<0||idxBoxes<0||idxScores<0||idxClasses<0){
            std::cerr<<"  Positional fallback\n";
            idxNumDets=0; idxBoxes=1; idxScores=2; idxClasses=3;
        }
        auto bsh=session_.GetOutputTypeInfo(idxBoxes)
                    .GetTensorTypeAndShapeInfo().GetShape();
        if(bsh.size()==3){
            if(bsh[2]==4){ boxesNHW_=true;  maxDet_=(int)bsh[1]; }
            else          { boxesNHW_=false; maxDet_=(int)bsh[2]; }
        } else if(bsh.size()==2){
            boxesNHW_=true; maxDet_=(int)bsh[0];
        }
    }
    else {
        outputFormat_=OutputFormat::RAW;
        std::cerr<<"[OnnxDetector] Format: RAW (default)\n";
    }
}

// Ã¢â€â‚¬Ã¢â€â‚¬ Letterbox preprocess Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
void OnnxDetector::preprocess(const cv::Mat& frame, std::vector<float>& blob)
{
    scale_=std::min((float)kInputSize/frame.cols,(float)kInputSize/frame.rows);
    int newW=(int)std::round(frame.cols*scale_);
    int newH=(int)std::round(frame.rows*scale_);
    padLeft_=(kInputSize-newW)/2; padTop_=(kInputSize-newH)/2;
    int padR=kInputSize-newW-padLeft_, padB=kInputSize-newH-padTop_;
    cv::Mat res,pad,rgb;
    cv::resize(frame,res,cv::Size(newW,newH),0,0,cv::INTER_LINEAR);
    cv::copyMakeBorder(res,pad,padTop_,padB,padLeft_,padR,
                       cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    cv::cvtColor(pad,rgb,cv::COLOR_BGR2RGB);
    cv::Mat bm=cv::dnn::blobFromImage(rgb,1.0/255.0,
                cv::Size(kInputSize,kInputSize),cv::Scalar(),false,false,CV_32F);
    blob.assign(bm.ptr<float>(),bm.ptr<float>()+bm.total());
}

static cv::Rect clampRect(float x1,float y1,float x2,float y2,int W,int H){
    x1=std::max(0.f,std::min(x1,(float)(W-1)));
    y1=std::max(0.f,std::min(y1,(float)(H-1)));
    x2=std::max(0.f,std::min(x2,(float)W));
    y2=std::max(0.f,std::min(y2,(float)H));
    return cv::Rect((int)x1,(int)y1,
                    std::max(1,(int)(x2-x1)),
                    std::max(1,(int)(y2-y1)));
}

// Ã¢â€â‚¬Ã¢â€â‚¬ NMS6: single tensor [1, N, 6]  each row = [x1,y1,x2,y2,conf,cls] Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
std::vector<Detection> OnnxDetector::postprocessNMS6(
        const cv::Mat& frame, const float* data, int numDets)
{
    const int W=frame.cols, H=frame.rows;

    // Auto-detect coordinate space once
    if(!spaceChecked_ && numDets>0){
        // If x2 or y2 > kInputSize, boxes are in original-image space
        float x2=data[2], y2=data[3];
        boxesAlreadyMapped_=(x2>(float)kInputSize || y2>(float)kInputSize);
        std::cerr<<"[OnnxDetector] NMS6 box space: "
                 <<(boxesAlreadyMapped_?"ORIGINAL-IMAGE":"640-LETTERBOX")<<"\n";
        std::cerr<<"  sample box[0]: "<<data[0]<<" "<<data[1]
                 <<" "<<data[2]<<" "<<data[3]
                 <<" conf="<<data[4]<<" cls="<<data[5]<<"\n";
        spaceChecked_=true;
    }

    std::vector<Detection> out;
    for(int i=0;i<numDets;++i){
        const float* r=data+i*6;
        float lx1=r[0], ly1=r[1], lx2=r[2], ly2=r[3];
        float conf=std::max(0.f,std::min(1.f,r[4]));
        int   cls =(int)r[5];

        if(conf<kConfThresh) continue;

        cv::Rect box;
        if(boxesAlreadyMapped_){
            box=clampRect(lx1,ly1,lx2,ly2,W,H);
        } else {
            float x1=(lx1-(float)padLeft_)/scale_;
            float y1=(ly1-(float)padTop_ )/scale_;
            float x2=(lx2-(float)padLeft_)/scale_;
            float y2=(ly2-(float)padTop_ )/scale_;
            box=clampRect(x1,y1,x2,y2,W,H);
        }
        out.push_back({box, cls, conf});
    }
    return out;
}

// Ã¢â€â‚¬Ã¢â€â‚¬ NMS4: 4 separate tensors Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
std::vector<Detection> OnnxDetector::postprocessNMS4(
        const cv::Mat& frame, std::vector<Ort::Value>& outputs)
{
    std::vector<Detection> out;
    int numDets=0;
    auto tp=outputs[idxNumDets].GetTensorTypeAndShapeInfo().GetElementType();
    if(tp==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        numDets=outputs[idxNumDets].GetTensorMutableData<int32_t>()[0];
    else if(tp==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        numDets=(int)outputs[idxNumDets].GetTensorMutableData<int64_t>()[0];
    else
        numDets=(int)outputs[idxNumDets].GetTensorMutableData<float>()[0];
    numDets=std::min(numDets,maxDet_);
    if(numDets<=0) return out;

    const float* boxes  =outputs[idxBoxes ].GetTensorMutableData<float>();
    const float* scores =outputs[idxScores].GetTensorMutableData<float>();
    auto& clsT=outputs[idxClasses];
    bool clsInt=(clsT.GetTensorTypeAndShapeInfo().GetElementType()
                 ==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    const int W=frame.cols,H=frame.rows;

    for(int i=0;i<numDets;++i){
        float conf=std::max(0.f,std::min(1.f,scores[i]));
        if(conf<kConfThresh) continue;
        int cls=clsInt?(int)clsT.GetTensorMutableData<int32_t>()[i]
                      :(int)clsT.GetTensorMutableData<float>()[i];
        float lx1,ly1,lx2,ly2;
        if(boxesNHW_){ lx1=boxes[i*4+0];ly1=boxes[i*4+1];lx2=boxes[i*4+2];ly2=boxes[i*4+3]; }
        else         { lx1=boxes[0*maxDet_+i];ly1=boxes[1*maxDet_+i];
                       lx2=boxes[2*maxDet_+i];ly2=boxes[3*maxDet_+i]; }
        float x1=(lx1-padLeft_)/scale_, y1=(ly1-padTop_)/scale_;
        float x2=(lx2-padLeft_)/scale_, y2=(ly2-padTop_)/scale_;
        out.push_back({clampRect(x1,y1,x2,y2,W,H),cls,conf});
    }
    return out;
}

// Ã¢â€â‚¬Ã¢â€â‚¬ RAW: [1, 84, 8400] Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
std::vector<Detection> OnnxDetector::postprocessRaw(
        const cv::Mat& frame, const float* data,
        const std::vector<int64_t>& shape)
{
    if(shape.size()!=3||shape[0]!=1||shape[1]<5){
        std::cerr<<"[OnnxDetector] RAW: unexpected shape\n"; return {};
    }
    int C=(int)shape[1],N=(int)shape[2],nc=C-4;
    int W=frame.cols,H=frame.rows;
    std::vector<cv::Rect> bx; std::vector<float> sc; std::vector<int> ci;
    for(int i=0;i<N;++i){
        float cx=data[0*N+i],cy=data[1*N+i],w=data[2*N+i],h=data[3*N+i];
        int bc=0; float bv=0.f;
        for(int c=0;c<nc;++c){float v=data[(4+c)*N+i];if(v>bv){bv=v;bc=c;}}
        if(bv<kConfThresh) continue;
        float x1=(cx-w*.5f-padLeft_)/scale_, y1=(cy-h*.5f-padTop_)/scale_;
        float x2=(cx+w*.5f-padLeft_)/scale_, y2=(cy+h*.5f-padTop_)/scale_;
        bx.push_back(clampRect(x1,y1,x2,y2,W,H));
        sc.push_back(bv); ci.push_back(bc);
    }
    std::vector<int> idx;
    if(!bx.empty()) cv::dnn::NMSBoxes(bx,sc,kConfThresh,kNmsThresh,idx);
    std::vector<Detection> out;
    for(int i:idx) out.push_back({bx[i],ci[i],sc[i]});
    return out;
}

// Ã¢â€â‚¬Ã¢â€â‚¬ detect() Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
std::vector<Detection> OnnxDetector::detect(cv::Mat& frame)
{
    if(frame.empty()) return {};
    std::vector<float> blob;
    preprocess(frame,blob);
    const int64_t inShape[4]={1,3,kInputSize,kInputSize};
    auto mi=Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU);
    auto iv=Ort::Value::CreateTensor<float>(mi,blob.data(),blob.size(),inShape,4);
    std::vector<const char*> op; for(auto& n:outputNames_) op.push_back(n.c_str());
    const char* in=inputName_.c_str();
    auto outs=session_.Run(Ort::RunOptions{nullptr},&in,&iv,1,op.data(),op.size());

    if(outputFormat_==OutputFormat::NMS4)
        return postprocessNMS4(frame,outs);

    if(outputFormat_==OutputFormat::NMS6){
        auto info=outs[0].GetTensorTypeAndShapeInfo();
        int numDets=(int)info.GetShape()[1];
        // Copy to float (handles fp16 automatic conversion by ORT)
        size_t tot=info.GetElementCount();
        std::vector<float> buf(tot);
        std::copy(outs[0].GetTensorMutableData<float>(),
                  outs[0].GetTensorMutableData<float>()+tot,buf.begin());
        return postprocessNMS6(frame,buf.data(),numDets);
    }

    // RAW
    auto info=outs[0].GetTensorTypeAndShapeInfo();
    auto sh=info.GetShape(); size_t tot=info.GetElementCount();
    std::vector<float> raw(tot);
    std::copy(outs[0].GetTensorMutableData<float>(),
              outs[0].GetTensorMutableData<float>()+tot,raw.begin());
    return postprocessRaw(frame,raw.data(),sh);
}


