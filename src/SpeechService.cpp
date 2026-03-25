// SpeechService.cpp
// Windows SAPI In-process Speech Recognition — dictation only, default audio, UTF-8.
#include "SpeechService.hpp"
#include "ResultsStore.h"
#include <iostream>
#include <string>
#include <cctype>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <sapi.h>
#endif

#ifdef _WIN32
// Create default audio input object from SAPI category (no sphelper.h).
static bool SetRecognizerDefaultAudioInput(ISpRecognizer* pRec) {
    ISpObjectTokenCategory* pCat = nullptr;
    IEnumSpObjectTokens*    pEnum = nullptr;
    ISpObjectToken*         pTok  = nullptr;

    if (FAILED(CoCreateInstance(CLSID_SpObjectTokenCategory, nullptr,
                                CLSCTX_ALL, IID_ISpObjectTokenCategory,
                                reinterpret_cast<void**>(&pCat))))
        return false;

    if (FAILED(pCat->SetId(SPCAT_AUDIOIN, FALSE))) {
        pCat->Release();
        return false;
    }
    if (FAILED(pCat->EnumTokens(nullptr, nullptr, &pEnum))) {
        pCat->Release();
        return false;
    }
    ULONG fetched = 0;
    pEnum->Next(1, &pTok, &fetched);
    pEnum->Release();
    pCat->Release();

    if (fetched == 0 || !pTok) {
        return false;
    }
    HRESULT hr = pRec->SetInput(pTok, TRUE);
    pTok->Release();
    return SUCCEEDED(hr);
}

// WCHAR* to UTF-8 std::string (no cast/sprintf).
static std::string WideToUtf8(const WCHAR* pwsz) {
    if (!pwsz || !*pwsz) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, pwsz, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, pwsz, -1, &out[0], len, nullptr, nullptr);
    return out;
}
#endif

namespace SmartGlasses {

SpeechService::SpeechService() {}
SpeechService::~SpeechService() { stopListening(); }

bool SpeechService::startListening(ResultCallback cb) {
    if (listening_) return true;
    callback_  = cb;
    listening_ = true;
    thread_ = std::thread(&SpeechService::listenLoop, this);
    return true;
}

void SpeechService::stopListening() {
    listening_ = false;
    if (thread_.joinable()) thread_.join();
}

std::string SpeechService::getLastResult() {
    std::lock_guard<std::mutex> lk(mutex_);
    return lastResult_;
}

void SpeechService::listenLoop() {
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "[STT] CoInitializeEx failed (hr=" << hr << ")\n";
        gResultsStore.setSAPIAvailable(false);
        gResultsStore.setSTTStatus("error");
        return;
    }

    ISpRecognizer*  pRec = nullptr;
    ISpRecoContext* pCtx = nullptr;
    ISpRecoGrammar* pGrm = nullptr;

    hr = CoCreateInstance(CLSID_SpInprocRecognizer, nullptr, CLSCTX_ALL,
                          IID_ISpRecognizer,
                          reinterpret_cast<void**>(&pRec));
    if (FAILED(hr)) {
        std::cerr << "[STT] SAPI not available — STT disabled (hr=" << hr << ")\n";
        if (hr == REGDB_E_CLASSNOTREG) {
            std::cerr << "[STT] SAPI not registered. Install Windows Speech Platform or enable Windows TTS voices in Settings.\n";
        }
        gResultsStore.setSAPIAvailable(false);
        gResultsStore.setSTTStatus("error");
        CoUninitialize();
        return;
    }

    // Explicitly set default audio input device.
    if (!SetRecognizerDefaultAudioInput(pRec)) {
        std::cerr << "[STT] SAPI: No audio input device found.\n";
    }

    hr = pRec->CreateRecoContext(&pCtx);
    if (FAILED(hr) || !pCtx) {
        if (pRec) pRec->Release();
        gResultsStore.setSAPIAvailable(false);
        gResultsStore.setSTTStatus("error");
        CoUninitialize();
        return;
    }

    pCtx->SetNotifyWin32Event();
    pCtx->SetInterest(SPFEI(SPEI_RECOGNITION), SPFEI(SPEI_RECOGNITION));
    HANDLE hEvent = pCtx->GetNotifyEventHandle();

    // Dictation grammar only (no command/rule grammar). Grammar ID 0.
    hr = pCtx->CreateGrammar(0, &pGrm);
    if (SUCCEEDED(hr) && pGrm) {
        hr = pGrm->LoadDictation(nullptr, SPLO_STATIC);
        if (SUCCEEDED(hr)) {
            pGrm->SetDictationState(SPRS_ACTIVE);
        } else {
            pGrm->Release();
            pGrm = nullptr;
        }
    }

    pRec->SetRecoState(SPRST_ACTIVE);
    gResultsStore.setSAPIAvailable(true);
    gResultsStore.setSTTActive(true);
    gResultsStore.setSTTStatus("listening");

    static const char* const kBlocklist[] = { "ao", "kitty", "the", "a", "um", "uh", nullptr };
    constexpr float kMinConfidence = 0.5f;

    while (listening_) {
        WaitForSingleObject(hEvent, 150);

        SPEVENT ev;
        ULONG   fetched = 0;
        while (SUCCEEDED(pCtx->GetEvents(1, &ev, &fetched)) && fetched > 0) {
            if (ev.eEventId != SPEI_RECOGNITION) continue;

            ISpRecoResult* pResult = reinterpret_cast<ISpRecoResult*>(ev.lParam);
            if (!pResult) continue;

            WCHAR* pwszText = nullptr;
            hr = pResult->GetText(SP_GETWHOLEPHRASE, SP_GETWHOLEPHRASE, TRUE, &pwszText, nullptr);
            if (FAILED(hr) || !pwszText) {
                pResult->Release();
                continue;
            }

            std::string text = WideToUtf8(pwszText);
            CoTaskMemFree(pwszText);
            pResult->Release();

            if (text.empty() || text.size() < 2u) continue;

            std::string lower = text;
            for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            bool blocked = false;
            for (const char* const* p = kBlocklist; *p; ++p) {
                if (lower == *p) { blocked = true; break; }
            }
            if (blocked) continue;
            (void)kMinConfidence;

            {
                std::lock_guard<std::mutex> lk(mutex_);
                lastResult_ = text;
            }
            gResultsStore.setSTTOriginal(text, "en");
            if (callback_) callback_(text);
        }
    }

    gResultsStore.setSTTActive(false);
    gResultsStore.setSTTStatus("disabled");
    if (pGrm) { pGrm->SetDictationState(SPRS_INACTIVE); pGrm->Release(); }
    if (pCtx) pCtx->Release();
    if (pRec) pRec->Release();
    CloseHandle(hEvent);
    CoUninitialize();
#else
    (void)0;  // STT not supported on non-Windows
#endif
}

} // namespace SmartGlasses