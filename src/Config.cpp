#include "Config.hpp"

namespace SmartGlasses {
    std::string Config::camera_url      = "";
    std::string Config::modelsDir       = "models";
    std::string Config::tessDataPrefix  = "models/tessdata";
    int         Config::cameraWidth     = 1280;
    int         Config::cameraHeight    = 720;
    bool        Config::TTS_ENABLED     = false;
    bool        Config::cudaEnabled     = true;
    float       Config::hudPositionY    = 0.0f;
    int         Config::activeLayer     = 0;
}