cmake_policy(SET CMP0146 OLD)
cmake_minimum_required(VERSION 3.20)
project(SmartGlassesHUD CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# â”€â”€ vcpkg toolchain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
set(ORT_DIR     "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-cuda")
set(IMGUI_DIR   "${CMAKE_SOURCE_DIR}/third_party/imgui")
set(WHISPER_DIR "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp")

# Optional self-heal for mismatched ORT GPU DLLs
option(CUDA_FIX "Copy onnxruntime*.dll from build/bin/Debug into third_party/onnxruntime-cuda/lib at configure time" OFF)
if(CUDA_FIX)
    set(_ort_debug_bin "${CMAKE_BINARY_DIR}/bin/Debug")
    if(EXISTS "${_ort_debug_bin}")
        file(GLOB _ort_debug_dlls "${_ort_debug_bin}/onnxruntime*.dll")
        if(_ort_debug_dlls)
            file(COPY ${_ort_debug_dlls} DESTINATION "${ORT_DIR}/lib")
            message(STATUS "CUDA_FIX: copied ONNX Runtime DLLs from ${_ort_debug_bin} to ${ORT_DIR}/lib")
        else()
            message(WARNING "CUDA_FIX: no onnxruntime*.dll found in ${_ort_debug_bin}")
        endif()
    else()
        message(WARNING "CUDA_FIX: Debug bin directory not found for CUDA_FIX: ${_ort_debug_bin}")
    endif()
endif()

# â”€â”€ Dear ImGui â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# â”€â”€ Whisper library (via upstream CMake) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

# Reduce warning noise from third-party code during MSVC builds.
if (MSVC)
    foreach(_t whisper ggml ggml-base ggml-cpu)
        if (TARGET ${_t})
            target_compile_options(${_t} PRIVATE /wd4244)
        endif()
    endforeach()
endif()

# â”€â”€ Find packages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
find_package(OpenCV        CONFIG REQUIRED)
find_package(glfw3         CONFIG REQUIRED)
find_package(glad          CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(Freetype             REQUIRED)
find_package(Tesseract     CONFIG REQUIRED)
find_package(Eigen3        CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog        CONFIG REQUIRED)
find_package(CURL          CONFIG REQUIRED)
find_package(cpr           CONFIG REQUIRED)
find_package(OpenSSL              REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(zstd          CONFIG REQUIRED)
find_package(CLI11         CONFIG REQUIRED)
find_package(fmt           CONFIG REQUIRED)
find_package(portaudio      CONFIG REQUIRED)
find_package(SampleRate     CONFIG REQUIRED)
find_package(nanoflann     CONFIG REQUIRED)

# â”€â”€ Main executable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
file(GLOB_RECURSE APP_SOURCES src/*.cpp)
list(FILTER APP_SOURCES EXCLUDE REGEX "src/object_detection_verify.cpp$")

add_executable(smart_glasses_hud
    ${APP_SOURCES}
    ${IMGUI_SOURCES}
)

# Output exe to bin/Release next to whisper/ggml DLLs (fixes STATUS_DLL_NOT_FOUND)
set_target_properties(smart_glasses_hud PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin/RelWithDebInfo"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin/MinSizeRel"
)

target_include_directories(smart_glasses_hud PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ORT_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${WHISPER_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(smart_glasses_hud PRIVATE
    ${OpenCV_LIBS}
    glfw
    glad::glad
    glm::glm
    Freetype::Freetype
    opengl32
    Tesseract::libtesseract
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    CURL::libcurl
    cpr::cpr
    OpenSSL::SSL OpenSSL::Crypto
    unofficial::sqlite3::sqlite3
    zstd::libzstd_shared
    CLI11::CLI11
    fmt::fmt
    portaudio
    nanoflann::nanoflann
    ${ORT_DIR}/lib/onnxruntime.lib
    whisper
)

target_compile_definitions(smart_glasses_hud PRIVATE
    MODELS_PATH="${CMAKE_SOURCE_DIR}/models"
    TESSDATA_PATH="${CMAKE_SOURCE_DIR}/models/tessdata"
    WHISPER_MODEL_PATH="${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/models/ggml-small.en.bin"
    # Enable optional subsystems that the SmartGlasses pipeline expects.
    # These were defined in the original VS project and are safe here because
    # we already link the corresponding libraries above.
    HAVE_TESSERACT=1
    HAVE_ONNXRUNTIME=1
    HAS_WHISPER=1
    HAVE_PORTAUDIO=1
    USE_LIBSAMPLERATE=1
    USE_CUDA
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
)

# â”€â”€ Copy ORT DLLs (GPU build) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If the local ONNX Runtime GPU directory is missing, skip the copy
# instead of failing the whole build. The exe can still run as long
# as the required DLLs are on PATH or next to the binary.
if (EXISTS "${ORT_DIR}/lib")
    set(ORT_GPU_DLLS
        "${ORT_DIR}/lib/onnxruntime.dll"
        "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll"
        "${ORT_DIR}/lib/onnxruntime_providers_shared.dll"
    )
    set(_ort_missing "")
    foreach(dll IN LISTS ORT_GPU_DLLS)
        if(NOT EXISTS "${dll}")
            list(APPEND _ort_missing "${dll}")
        endif()
    endforeach()
    if(_ort_missing)
        message(FATAL_ERROR
            "ORT GPU DLLs missing in ${ORT_DIR}/lib:\n  ${_ort_missing}")
    endif()

    add_custom_command(TARGET smart_glasses_hud POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ONNX Runtime GPU DLLs..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:smart_glasses_hud>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll" "$<TARGET_FILE_DIR:smart_glasses_hud>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/onnxruntime_providers_shared.dll" "$<TARGET_FILE_DIR:smart_glasses_hud>/"
    )
else()
    message(WARNING "ORT_DIR/lib not found (${ORT_DIR}/lib) - skipping copy of ONNX Runtime DLLs")
endif()

# â”€â”€ Copy Run scripts (with CUDA/cuDNN PATH fix) to output â”€â”€â”€â”€â”€â”€â”€
if(EXISTS "${CMAKE_SOURCE_DIR}/Run_SmartGlassesHUD.ps1")
    add_custom_command(TARGET smart_glasses_hud POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/Run_SmartGlassesHUD.ps1"
            "$<TARGET_FILE_DIR:smart_glasses_hud>/Run_SmartGlassesHUD.ps1"
    )
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/Run_SmartGlassesHUD.bat")
    add_custom_command(TARGET smart_glasses_hud POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/Run_SmartGlassesHUD.bat"
            "$<TARGET_FILE_DIR:smart_glasses_hud>/Run_SmartGlassesHUD.bat"
    )
endif()

# â”€â”€ CUDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
find_package(CUDA QUIET)

# Configure-time CUDA/cuDNN/ORT compatibility guard
set(ORT_EXPECTED_CUDNN_MAJOR "")
set(_ort_cudnn_version_file "${ORT_DIR}/ORT_CUDNN_VERSION.txt")
if(EXISTS "${_ort_cudnn_version_file}")
    file(READ "${_ort_cudnn_version_file}" ORT_EXPECTED_CUDNN_MAJOR)
    string(STRIP "${ORT_EXPECTED_CUDNN_MAJOR}" ORT_EXPECTED_CUDNN_MAJOR)
else()
    # Default expectation for the bundled ORT in this repo
    set(ORT_EXPECTED_CUDNN_MAJOR "9")
endif()

if(CUDA_FOUND)
    message(STATUS "CUDA ${CUDA_VERSION} - GPU ON")
    target_compile_definitions(smart_glasses_hud PRIVATE USE_CUDA=1)
    target_link_libraries(smart_glasses_hud PRIVATE ${CUDA_LIBRARIES})

    if(WIN32)
        set(CUDA_BIN "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        if(NOT EXISTS "${CUDA_BIN}/cudart64_12.dll")
            message(FATAL_ERROR
                "Expected CUDA 12 runtime (cudart64_12.dll) in ${CUDA_BIN} but it was not found.")
        endif()

        # Detect CUDA major version from CUDA_VERSION if available
        if(CUDA_VERSION)
            string(REGEX MATCH "^[0-9]+" CUDA_VERSION_MAJOR "${CUDA_VERSION}")
            message(STATUS "Detected CUDA major version: ${CUDA_VERSION_MAJOR}")
        endif()

        # Copy CUDA DLLs needed by ORT GPU
        message(STATUS "ORT GPU: copying CUDA DLLs from ${CUDA_BIN}")
        add_custom_command(TARGET smart_glasses_hud POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying CUDA runtime DLLs for ORT GPU..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cudart64_12.dll"        "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cublas64_12.dll"        "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cublasLt64_12.dll"      "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cufft64_11.dll"         "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/curand64_10.dll"        "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cusolver64_11.dll"      "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/cusparse64_12.dll"      "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/nvrtc64_120_0.dll"      "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/nvrtc-builtins64_126.dll" "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_BIN}/nvJitLink_120_0.dll"    "$<TARGET_FILE_DIR:smart_glasses_hud>/"
        )

        # Locate cuDNN and detect its major version from DLL name
        set(CUDNN_CANDIDATES
            "C:/Program Files/NVIDIA/CUDNN/v9.19/bin/12.9/x64"
            "C:/Program Files/NVIDIA/CUDNN/v9.3/bin/12.6/x64"
            "C:/Program Files/NVIDIA/CUDNN/v9.2/bin/12.4/x64"
        )
        set(CUDNN_BIN "")
        foreach(C ${CUDNN_CANDIDATES})
            if(EXISTS "${C}")
                set(CUDNN_BIN "${C}")
                break()
            endif()
        endforeach()
        if(CUDNN_BIN STREQUAL "")
            file(GLOB CUDNN_GLOB "C:/Program Files/NVIDIA/CUDNN/*/bin/12*/x64")
            list(LENGTH CUDNN_GLOB CUDNN_LEN)
            if(CUDNN_LEN GREATER 0)
                list(GET CUDNN_GLOB 0 CUDNN_BIN)
            endif()
        endif()

        if(NOT CUDNN_BIN)
            message(FATAL_ERROR "cuDNN not found under C:/Program Files/NVIDIA/CUDNN - required for ORT GPU.")
        endif()

        # Determine installed cuDNN major version from cudnn64_X.dll
        file(GLOB _cudnn_name_candidates "${CUDNN_BIN}/cudnn64_*.dll")
        if(NOT _cudnn_name_candidates)
            message(FATAL_ERROR "No cudnn64_*.dll found in ${CUDNN_BIN} - cuDNN installation incomplete.")
        endif()
        list(GET _cudnn_name_candidates 0 _cudnn_fullpath)
        get_filename_component(_cudnn_filename "${_cudnn_fullpath}" NAME)
        string(REGEX MATCH "cudnn64_([0-9]+)\\.dll" _cudnn_match "${_cudnn_filename}")
        set(CUDNN_MAJOR "${CMAKE_MATCH_1}")
        message(STATUS "Detected cuDNN major version: ${CUDNN_MAJOR} (from ${_cudnn_filename})")

        # Compare with ORT's expected cuDNN major version
        if(ORT_EXPECTED_CUDNN_MAJOR AND CUDNN_MAJOR AND NOT ORT_EXPECTED_CUDNN_MAJOR STREQUAL CUDNN_MAJOR)
            message(FATAL_ERROR
                "ONNX Runtime in ${ORT_DIR} expects cuDNN ${ORT_EXPECTED_CUDNN_MAJOR} "
                "but detected cuDNN ${CUDNN_MAJOR} in ${CUDNN_BIN}. "
                "Please update ONNX Runtime or install matching cuDNN.")
        endif()

        if(NOT EXISTS "${CUDNN_BIN}/cudnn64_${CUDNN_MAJOR}.dll")
            message(FATAL_ERROR "cudnn64_${CUDNN_MAJOR}.dll not found in ${CUDNN_BIN}")
        endif()

        message(STATUS "ORT GPU: copying cuDNN DLLs from ${CUDNN_BIN}")

        # Collect cuDNN CNN / training DLLs that actually exist.
        # Some cuDNN builds may omit training DLLs; in that case we skip them
        # instead of failing the whole build, since ORT can still run inference.
        set(CUDNN_OPTIONAL_DLLS "")
        if(EXISTS "${CUDNN_BIN}/cudnn_cnn64_${CUDNN_MAJOR}.dll")
            list(APPEND CUDNN_OPTIONAL_DLLS "${CUDNN_BIN}/cudnn_cnn64_${CUDNN_MAJOR}.dll")
        endif()
        if(EXISTS "${CUDNN_BIN}/cudnn_cnn_train64_${CUDNN_MAJOR}.dll")
            list(APPEND CUDNN_OPTIONAL_DLLS "${CUDNN_BIN}/cudnn_cnn_train64_${CUDNN_MAJOR}.dll")
        endif()

        add_custom_command(TARGET smart_glasses_hud POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying cuDNN DLLs for ORT GPU..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDNN_BIN}/cudnn64_${CUDNN_MAJOR}.dll"        "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDNN_BIN}/cudnn_ops64_${CUDNN_MAJOR}.dll"    "$<TARGET_FILE_DIR:smart_glasses_hud>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDNN_OPTIONAL_DLLS} "$<TARGET_FILE_DIR:smart_glasses_hud>/"
        )
    endif()
else()
    message(WARNING "CUDA not found - CPU fallback")
endif()

# â”€â”€ Verify tool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
add_executable(object_detection_verify
    tools/object_detection_verify.cpp
    src/object/object_detector.cpp
    src/object/object_store.cpp
    src/utils/ort_session.cpp
    src/utils/image_utils.cpp
)
target_include_directories(object_detection_verify PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ORT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(object_detection_verify PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    ${ORT_DIR}/lib/onnxruntime.lib
)
target_compile_definitions(object_detection_verify PRIVATE
    MODELS_PATH="${CMAKE_SOURCE_DIR}/models"
    NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
)




# Auto-copy Python scripts and models to output dir after every build
add_custom_command(TARGET smart_glasses_hud POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/scripts"
        "$<TARGET_FILE_DIR:smart_glasses_hud>/scripts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/models"
        "$<TARGET_FILE_DIR:smart_glasses_hud>/models"
    COMMENT "Copying scripts and models to output directory"
)
