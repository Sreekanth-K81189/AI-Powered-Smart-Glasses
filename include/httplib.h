#pragma once

// Minimal HTTP client shim with a cpp-httplib compatible surface
// for the subset used by TranslationEngine (Client::Get/Post + timeouts).
//
// This is intentionally lightweight and Windows-only (WinHTTP).
// It does NOT provide HTTPS/OpenSSL support.

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <winhttp.h>
#  pragma comment(lib, "winhttp.lib")
#else
#  error "httplib.h shim is Windows-only in this project"
#endif

#include <memory>
#include <string>
#include <vector>

namespace httplib {

struct Response {
  int status = 0;
  std::string body;
};

class Client {
public:
  Client(const std::string& host, int port) : host_(host), port_(port) {}

  void set_connection_timeout(int seconds) { conn_timeout_ms_ = seconds * 1000; }
  void set_read_timeout(int seconds) { read_timeout_ms_ = seconds * 1000; }

  std::shared_ptr<Response> Get(const char* path) {
    return request(L"GET", path, nullptr, 0, nullptr);
  }

  std::shared_ptr<Response> Post(const char* path,
                                 const std::string& body,
                                 const char* content_type) {
    return request(L"POST", path, body.data(), (DWORD)body.size(), content_type);
  }

private:
  std::shared_ptr<Response> request(const wchar_t* method,
                                    const char* path,
                                    const void* body,
                                    DWORD body_len,
                                    const char* content_type) {
    auto out = std::make_shared<Response>();

    HINTERNET hSession = WinHttpOpen(L"SmartGlassesHUD/1.0",
                                    WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                    WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return nullptr;

    WinHttpSetTimeouts(hSession, conn_timeout_ms_, conn_timeout_ms_, read_timeout_ms_, read_timeout_ms_);

    std::wstring whost(host_.begin(), host_.end());
    HINTERNET hConnect = WinHttpConnect(hSession, whost.c_str(), (INTERNET_PORT)port_, 0);
    if (!hConnect) {
      WinHttpCloseHandle(hSession);
      return nullptr;
    }

    std::wstring wpath;
    for (const char* p = path; *p; ++p) wpath.push_back((wchar_t)(unsigned char)*p);

    HINTERNET hRequest = WinHttpOpenRequest(
      hConnect, method, wpath.c_str(),
      nullptr, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES,
      0);
    if (!hRequest) {
      WinHttpCloseHandle(hConnect);
      WinHttpCloseHandle(hSession);
      return nullptr;
    }

    std::wstring headers;
    if (content_type && *content_type) {
      std::string h = std::string("Content-Type: ") + content_type + "\r\n";
      headers.assign(h.begin(), h.end());
    }

    BOOL ok = WinHttpSendRequest(
      hRequest,
      headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(),
      headers.empty() ? 0 : (DWORD)-1L,
      (LPVOID)body,
      body_len,
      body_len,
      0);
    if (!ok) {
      WinHttpCloseHandle(hRequest);
      WinHttpCloseHandle(hConnect);
      WinHttpCloseHandle(hSession);
      return nullptr;
    }

    ok = WinHttpReceiveResponse(hRequest, nullptr);
    if (!ok) {
      WinHttpCloseHandle(hRequest);
      WinHttpCloseHandle(hConnect);
      WinHttpCloseHandle(hSession);
      return nullptr;
    }

    DWORD status = 0;
    DWORD statusSize = sizeof(status);
    WinHttpQueryHeaders(hRequest,
                        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &statusSize, WINHTTP_NO_HEADER_INDEX);
    out->status = (int)status;

    std::string resp;
    for (;;) {
      DWORD avail = 0;
      if (!WinHttpQueryDataAvailable(hRequest, &avail)) break;
      if (avail == 0) break;
      std::vector<char> buf(avail);
      DWORD read = 0;
      if (!WinHttpReadData(hRequest, buf.data(), avail, &read)) break;
      if (read == 0) break;
      resp.append(buf.data(), buf.data() + read);
    }
    out->body = std::move(resp);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return out;
  }

  std::string host_;
  int port_ = 0;
  int conn_timeout_ms_ = 2000;
  int read_timeout_ms_ = 5000;
};

} // namespace httplib

