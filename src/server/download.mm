// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The Hugging Face Hub, natively (download.h): the API listing, byte
// ranges and the HTTP transfers under them, with NSURLSession (one ephemeral
// session per transfer, as src/vision/fetch.mm does for images). Redirects
// to the Hub's CDN are followed; the token never leaves the endpoint's host.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "../util/json.h"
#include "../util/paths.h"
#include "download.h"
#include "models.h"
#include "version.h"

namespace {

using espetero::hub::HttpResponse;

std::string str(NSString* s) { return s ? std::string([s UTF8String] ?: "") : std::string(); }
NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()] ?: @""; }

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string env(const char* k) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string();
}

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  return s.substr(a);
}

std::string expand_home(std::string s) {
  if (s == "~" || s.rfind("~/", 0) == 0) s = espetero::paths::home() + s.substr(1);
  return s;
}

std::string host_of(NSURL* u) { return lower(str(u.host)); }

// Shared by the caller and the session's delegate queue (which may deliver a
// late callback after the caller has returned).
struct State {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  bool stopped = false;  // a callback stopped the transfer
  HttpResponse res;
  int redirects = 0;
  std::string token_host;  // the endpoint's host: the only one that gets the token
  bool https = true;       // the first URL was https: no redirect to http
  std::function<bool(const HttpResponse&)> on_response;
  std::function<bool(const char*, size_t)> on_data;
  void fail(const std::string& e) {
    std::lock_guard<std::mutex> lk(mu);
    if (res.error.empty()) res.error = e;
  }
};

}  // namespace

@interface EspeteroHubTransfer : NSObject <NSURLSessionDataDelegate>
@end

@implementation EspeteroHubTransfer {
 @public
  std::shared_ptr<State> st;
}

- (void)URLSession:(NSURLSession*)session
              dataTask:(NSURLSessionDataTask*)task
    didReceiveResponse:(NSURLResponse*)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))handler {
  State& s = *st;
  NSHTTPURLResponse* http = [response isKindOfClass:[NSHTTPURLResponse class]] ? (NSHTTPURLResponse*)response : nil;
  if (!http) {
    s.fail("not an HTTP response");
    handler(NSURLSessionResponseCancel);
    return;
  }
  HttpResponse r;
  r.status = int(http.statusCode);
  NSDictionary* h = http.allHeaderFields;
  for (id k in h) r.headers[lower(str([k description]))] = str([h[k] description]);
  bool go = true;
  if (s.on_response) go = s.on_response(r);
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.res.status = r.status;
    s.res.headers = std::move(r.headers);
    if (!go) s.stopped = true;
  }
  handler(go ? NSURLSessionResponseAllow : NSURLSessionResponseCancel);
}

- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task didReceiveData:(NSData*)data {
  State& s = *st;
  if (s.stopped) return;
  __block bool go = true;
  [data enumerateByteRangesUsingBlock:^(const void* bytes, NSRange range, BOOL* stop) {
    if (s.on_data && !s.on_data(static_cast<const char*>(bytes), range.length)) {
      go = false;
      *stop = YES;
    }
  }];
  if (!go) {
    {
      std::lock_guard<std::mutex> lk(s.mu);
      s.stopped = true;
    }
    [task cancel];
  }
}

- (void)URLSession:(NSURLSession*)session
                          task:(NSURLSessionTask*)task
    willPerformHTTPRedirection:(NSHTTPURLResponse*)response
                    newRequest:(NSURLRequest*)request
             completionHandler:(void (^)(NSURLRequest*))handler {
  State& s = *st;
  const std::string scheme = lower(str(request.URL.scheme));
  std::string why;
  if (++s.redirects > 10) why = "more than 10 redirects";
  else if (scheme != "https" && (s.https || scheme != "http")) why = "redirected to a non-https URL";
  if (!why.empty()) {
    s.fail(why);
    handler(nil);
    [task cancel];
    return;
  }
  // The CDN gets a signed URL: the token stays with the endpoint's host.
  NSMutableURLRequest* r = [request mutableCopy];
  if (host_of(request.URL) != s.token_host) [r setValue:nil forHTTPHeaderField:@"Authorization"];
  handler(r);
}

- (void)URLSession:(NSURLSession*)session
                   task:(NSURLSessionTask*)task
    didReceiveChallenge:(NSURLAuthenticationChallenge*)challenge
      completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition, NSURLCredential*))handler {
  if ([challenge.protectionSpace.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust])
    handler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
  else
    handler(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task didCompleteWithError:(NSError*)error {
  State& s = *st;
  if (error && !s.stopped) {
    if (error.code == NSURLErrorTimedOut) s.fail("no data for too long (timed out)");
    else s.fail(str(error.localizedDescription));
  }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.done = true;
  }
  s.cv.notify_all();
}

@end

namespace espetero::hub {

std::string HttpResponse::header(const std::string& lower_name) const {
  auto it = headers.find(lower_name);
  return it == headers.end() ? std::string() : it->second;
}

const RepoFile* RepoInfo::find(const std::string& path) const {
  for (const RepoFile& f : files)
    if (f.path == path) return &f;
  return nullptr;
}

std::string endpoint() {
  std::string e = trim(env("HF_ENDPOINT"));
  while (!e.empty() && e.back() == '/') e.pop_back();
  return e.empty() ? "https://huggingface.co" : e;
}

std::string token() {
  for (const char* k : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
    if (std::string t = trim(env(k)); !t.empty()) return t;
  std::string path = expand_home(trim(env("HF_TOKEN_PATH")));
  if (path.empty()) {
    std::string home = expand_home(trim(env("HF_HOME")));
    if (home.empty()) {
      const std::string xdg = expand_home(trim(env("XDG_CACHE_HOME")));
      home = (xdg.empty() ? paths::home() + "/.cache" : xdg) + "/huggingface";
    }
    path = home + "/token";
  }
  std::ifstream f(path);
  std::stringstream ss;
  if (f) ss << f.rdbuf();
  return trim(ss.str());
}

HttpResponse http_stream(const std::string& url, const Headers& headers,
                         const std::function<bool(const HttpResponse&)>& on_response,
                         const std::function<bool(const char*, size_t)>& on_data, double idle_s) {
  @autoreleasepool {
    auto st = std::make_shared<State>();
    st->on_response = on_response;
    st->on_data = on_data;
    NSURL* u = [NSURL URLWithString:ns(url)];
    const std::string scheme = u ? lower(str(u.scheme)) : std::string();
    if (!u || (scheme != "https" && scheme != "http") || host_of(u).empty()) {
      st->res.error = "not a valid URL: " + url;
      return st->res;
    }
    st->https = scheme == "https";
    NSURL* ep = [NSURL URLWithString:ns(endpoint())];
    st->token_host = ep ? host_of(ep) : std::string();
    NSURLSessionConfiguration* cfg = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    cfg.timeoutIntervalForRequest = idle_s;
    cfg.timeoutIntervalForResource = 7 * 24 * 3600;
    cfg.HTTPCookieStorage = nil;
    cfg.HTTPShouldSetCookies = NO;
    cfg.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
    cfg.URLCache = nil;
    cfg.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
    cfg.URLCredentialStorage = nil;
    cfg.waitsForConnectivity = NO;
    cfg.HTTPAdditionalHeaders = @{@"User-Agent" : [NSString stringWithFormat:@"%s/%s", espetero::kProduct, espetero::kVersion]};
    NSMutableURLRequest* req = [NSMutableURLRequest requestWithURL:u];
    for (const auto& [k, v] : headers) [req setValue:ns(v) forHTTPHeaderField:ns(k)];
    if (host_of(u) == st->token_host)
      if (const std::string t = token(); !t.empty()) [req setValue:ns("Bearer " + t) forHTTPHeaderField:@"Authorization"];
    EspeteroHubTransfer* d = [[EspeteroHubTransfer alloc] init];
    d->st = st;
    NSOperationQueue* q = [[NSOperationQueue alloc] init];
    q.maxConcurrentOperationCount = 1;
    NSURLSession* session = [NSURLSession sessionWithConfiguration:cfg delegate:d delegateQueue:q];
    NSURLSessionDataTask* task = [session dataTaskWithRequest:req];
    [task resume];
    {
      std::unique_lock<std::mutex> lk(st->mu);
      st->cv.wait(lk, [&] { return st->done; });
    }
    [session invalidateAndCancel];  // releases the delegate
    std::lock_guard<std::mutex> lk(st->mu);
    return st->res;
  }
}

HttpResponse http_get(const std::string& url, const Headers& headers, size_t max_body, double idle_s) {
  std::string body;
  HttpResponse r = http_stream(url, headers, nullptr,
                               [&](const char* p, size_t n) {
                                 const size_t take = std::min(n, max_body - body.size());
                                 body.append(p, take);
                                 return body.size() < max_body;
                               },
                               idle_s);
  r.body = std::move(body);
  return r;
}

namespace {

// Percent-encodes a path (keeping '/').
std::string url_path(const std::string& p, bool keep_slash = true) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : p) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (keep_slash && c == '/')) o += char(c);
    else o += '%', o += hex[c >> 4], o += hex[c & 15];
  }
  return o;
}

bool is_hex(const std::string& s, size_t n) {
  if (s.size() != n) return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

// A repository path that stays inside the snapshot directory.
bool safe_path(const std::string& p) {
  if (p.empty() || p[0] == '/' || p.size() > 1024) return false;
  size_t a = 0;
  while (a <= p.size()) {
    size_t b = p.find('/', a);
    if (b == std::string::npos) b = p.size();
    const std::string seg = p.substr(a, b - a);
    if (seg.empty() || seg == "." || seg == "..") return false;
    a = b + 1;
  }
  for (unsigned char c : p)
    if (c < 0x20 || c == 0x7f || c == '\\') return false;
  return true;
}

// The Hub's reason for a failed request, for the user.
std::string hub_error(const std::string& repo, const std::string& what, const HttpResponse& r) {
  if (!r.error.empty())
    return "cannot reach " + endpoint() + " for " + repo + ": " + r.error +
           " (with no network, --offline runs what is in the Hugging Face cache)";
  const std::string code = r.header("x-error-code");
  std::string msg = r.header("x-error-message");
  if (msg.empty()) {
    try {
      msg = json::parse(r.body).get_string("error", "");
    } catch (const std::exception&) {
    }
  }
  const bool have = !token().empty();
  const std::string page = endpoint() + "/" + repo;
  const std::string how = "set HF_TOKEN to a read token (" + endpoint() +
                          "/settings/tokens) or save it in ~/.cache/huggingface/token";
  if (code == "GatedRepo")
    return repo + " is gated: accept its conditions at " + page +
           (have ? " with the account of your token, then try again" : ", then " + how);
  if (code == "RepoNotFound" || ((r.status == 401 || r.status == 404) && code.empty() && what == "listing"))
    return repo + " was not found on " + endpoint() + (have ? "" : " (a private repository needs a token: " + how + ")");
  if (code == "RevisionNotFound" || code == "EntryNotFound")
    return repo + ": " + (msg.empty() ? what + " not found" : msg);
  if (r.status == 401 || r.status == 403)
    return repo + ": access denied (HTTP " + std::to_string(r.status) + (msg.empty() ? "" : ": " + msg) + ")" +
           (have ? "" : "; " + how);
  if (r.status == 429) return repo + ": the Hub limits the request rate (HTTP 429); try again in a few minutes" +
                              (have ? "" : " or " + how);
  return repo + ": " + endpoint() + " answered HTTP " + std::to_string(r.status) + " to the " + what +
         (msg.empty() ? "" : ": " + msg);
}

}  // namespace

std::string resolve_url(const std::string& repo, const std::string& revision, const std::string& path) {
  return endpoint() + "/" + repo + "/resolve/" + url_path(revision, false) + "/" + url_path(path);
}

RepoInfo repo_info(const std::string& repo, const std::string& revision) {
  const std::string url = endpoint() + "/api/models/" + repo + "/revision/" + url_path(revision, false) + "?blobs=true";
  const HttpResponse r = http_get(url, {{"Accept", "application/json"}}, size_t(64) << 20);
  if (!r.error.empty() || r.status != 200) throw Error(hub_error(repo, "listing", r));
  RepoInfo info;
  info.repo = repo;
  info.revision = revision;
  try {
    const json::Value v = json::parse(r.body);
    info.commit = v.get_string("sha", "");
    if (const json::Value* g = v.get("gated"); g && g->is_string()) info.gated = g->as_string();
    if (const json::Value* sib = v.get("siblings"); sib && sib->is_array())
      for (const json::Value& s : sib->as_array()) {
        RepoFile f;
        f.path = s.get_string("rfilename", "");
        f.size = s.get_int("size", 0);
        f.blob = s.get_string("blobId", "");
        if (const json::Value* lfs = s.get("lfs"); lfs && lfs->is_object()) {
          f.lfs = true;
          f.blob = lfs->get_string("sha256", "");
          f.size = lfs->get_int("size", f.size);
        }
        // Names that would leave the snapshot or the blobs directory are not files of a model.
        if (!safe_path(f.path) || !(is_hex(f.blob, 40) || is_hex(f.blob, 64))) continue;
        f.blob = lower(f.blob);
        info.files.push_back(std::move(f));
      }
  } catch (const std::exception& e) {
    throw Error(repo + ": unexpected answer from " + endpoint() + " (" + e.what() + ")");
  }
  if (!is_hex(info.commit, 40)) throw Error(repo + ": unexpected answer from " + endpoint() + " (no commit hash)");
  info.commit = lower(info.commit);
  return info;
}

// ---------------------------------------------------------------- downloads
namespace {

using Clock = std::chrono::steady_clock;

// The digest a blob's name is: sha256 of an LFS file, git's blob id
// (sha1 of "blob <size>\0" + content) of any other.
class Digest {
 public:
  Digest(bool sha256, int64_t size) : sha256_(sha256), size_(size) { reset(); }
  void reset() {
    if (sha256_) {
      CC_SHA256_Init(&s256_);
      return;
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // git names blobs by sha1
    CC_SHA1_Init(&s1_);
    const std::string head = "blob " + std::to_string(size_) + std::string(1, '\0');
    CC_SHA1_Update(&s1_, head.data(), CC_LONG(head.size()));
#pragma clang diagnostic pop
  }
  void update(const char* p, size_t n) {
    while (n) {
      const CC_LONG k = CC_LONG(std::min<size_t>(n, size_t(1) << 30));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      if (sha256_) CC_SHA256_Update(&s256_, p, k);
      else CC_SHA1_Update(&s1_, p, k);
#pragma clang diagnostic pop
      p += k;
      n -= k;
    }
  }
  std::string hex() {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    size_t n = CC_SHA256_DIGEST_LENGTH;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if (sha256_) CC_SHA256_Final(d, &s256_);
    else CC_SHA1_Final(d, &s1_), n = CC_SHA1_DIGEST_LENGTH;
#pragma clang diagnostic pop
    static const char* x = "0123456789abcdef";
    std::string o;
    for (size_t i = 0; i < n; ++i) o += x[d[i] >> 4], o += x[d[i] & 15];
    return o;
  }

 private:
  bool sha256_;
  int64_t size_;
  CC_SHA256_CTX s256_;
  CC_SHA1_CTX s1_;
};

bool is_file(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

int64_t size_of(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) ? int64_t(st.st_size) : -1;
}

void mkdirs(const std::string& dir) {
  for (size_t i = 1; i <= dir.size(); ++i)
    if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), 0755);
}

std::string dirname_of(const std::string& p) { return p.substr(0, p.find_last_of('/')); }

using models::size_text;

std::string clock_text(double s) {
  char b[32];
  if (s >= 3600) std::snprintf(b, sizeof b, "%dh%02d", int(s / 3600), int(s / 60) % 60);
  else std::snprintf(b, sizeof b, "%d:%02d", int(s / 60), int(s) % 60);
  return b;
}

// The progress line on stderr: rewritten in place on a terminal, a line
// every 30 s otherwise.
class Progress {
 public:
  Progress(int64_t total, int64_t done, bool quiet) : total_(total), done_(done), start_done_(done), tty_(::isatty(2)), quiet_(quiet) {
    start_ = last_ = Clock::now();
  }
  void add(int64_t n, const std::string& file) {
    std::lock_guard<std::mutex> lk(mu_);
    done_ += n;
    file_ = file;
    if (quiet_) return;
    const auto now = Clock::now();
    if (now - last_ < (tty_ ? std::chrono::milliseconds(250) : std::chrono::milliseconds(30000))) return;
    last_ = now;
    print(now, false);
  }
  void finish(bool ok) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!quiet_ && (printed_ || ok)) print(Clock::now(), true);
  }

 private:
  void print(Clock::time_point now, bool last) {
    const double s = std::chrono::duration<double>(now - start_).count();
    const double rate = s > 0 ? double(done_ - start_done_) / s : 0;
    std::string line = "[espetero] " + size_text(done_) + " of " + size_text(total_) + " (" +
                       std::to_string(total_ > 0 ? int(100.0 * double(done_) / double(total_)) : 100) + " %)";
    if (rate > 0 && s >= 0.5) {
      line += ", " + size_text(int64_t(rate)) + "/s";
      if (!last && done_ < total_) line += ", " + clock_text(double(total_ - done_) / rate) + " left";
    }
    if (last) line += " in " + clock_text(s);
    else if (!file_.empty()) line += ": " + file_.substr(file_.find_last_of('/') + 1);
    if (tty_) std::fprintf(stderr, "\r%s\x1b[K%s", line.c_str(), last ? "\n" : "");
    else std::fprintf(stderr, "%s\n", line.c_str());
    printed_ = true;
  }
  std::mutex mu_;
  int64_t total_, done_, start_done_;
  bool tty_, quiet_, printed_ = false;
  Clock::time_point start_, last_;
  std::string file_;
};

bool write_all(int fd, const char* p, size_t n) {
  while (n) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return false;
    p += w;
    n -= size_t(w);
  }
  return true;
}

// blobs/<blob>.incomplete, resumed where an earlier attempt stopped, then
// verified and renamed to blobs/<blob>. The caller holds the blob's lock.
void fetch_blob(const RepoInfo& info, const RepoFile& f, const std::string& blob, Progress& prog,
                const std::atomic<bool>& stop) {
  const std::string part = blob + ".incomplete";
  const int fd = ::open(part.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) throw Error("cannot write " + part + ": " + std::strerror(errno));
  struct Close {
    int fd;
    ~Close() { ::close(fd); }
  } close_fd{fd};
  int64_t have = ::lseek(fd, 0, SEEK_END);
  Digest dg(f.lfs, f.size);
  if (have > f.size) {  // not this blob's start: from the beginning (the plan counted f.size of it)
    prog.add(-f.size, f.path);
    if (::ftruncate(fd, 0) != 0) throw Error("cannot write " + part + ": " + std::strerror(errno));
    have = 0;
  }
  if (have > 0) {  // what an earlier attempt wrote counts toward the digest
    std::string buf(size_t(8) << 20, '\0');
    for (int64_t off = 0; off < have;) {
      const ssize_t n = ::pread(fd, buf.data(), size_t(std::min<int64_t>(int64_t(buf.size()), have - off)), off);
      if (n <= 0) throw Error("cannot read " + part + ": " + std::strerror(errno));
      dg.update(buf.data(), size_t(n));
      off += n;
    }
  }
  for (int attempt = 0; have < f.size; ++attempt) {
    if (stop) throw Error("stopped");
    Headers h = {{"Accept-Encoding", "identity"}};
    if (have > 0) h.push_back({"Range", "bytes=" + std::to_string(have) + "-"});
    std::string fatal, why;
    const HttpResponse r = http_stream(
        resolve_url(info.repo, info.commit, f.path), h,
        [&](const HttpResponse& res) {
          if (res.status == 206) {
            if (res.header("content-range").rfind("bytes " + std::to_string(have) + "-", 0) == 0) return true;
            why = "the server sent another range (" + res.header("content-range") + ")";
            return false;
          }
          if (res.status != 200) return false;  // an error: its body is not the file
          if (have > 0) {                       // the whole file instead of the rest: start over
            prog.add(-have, f.path);
            if (::ftruncate(fd, 0) != 0 || ::lseek(fd, 0, SEEK_SET) != 0) {
              fatal = "cannot write " + part + ": " + std::strerror(errno);
              return false;
            }
            have = 0;
            dg.reset();
          }
          return true;
        },
        [&](const char* p, size_t n) {
          if (stop) return false;
          if (have + int64_t(n) > f.size) {
            why = "the server sent more than the file's " + std::to_string(f.size) + " bytes";
            return false;
          }
          if (!write_all(fd, p, n)) {
            fatal = "cannot write " + part + ": " + std::strerror(errno);
            return false;
          }
          dg.update(p, n);
          have += int64_t(n);
          prog.add(int64_t(n), f.path);
          return true;
        },
        60);
    if (!fatal.empty()) throw Error(fatal);
    if (stop) throw Error("stopped");
    if (have == f.size) break;
    if (r.error.empty() && r.status != 200 && r.status != 206 && r.status != 429 && r.status < 500)
      throw Error(hub_error(info.repo, f.path, r));
    if (why.empty()) why = !r.error.empty() ? r.error : r.status == 200 || r.status == 206 ? "the connection ended early"
                                                                                       : "HTTP " + std::to_string(r.status);
    if (attempt >= 5)
      throw Error(info.repo + ": " + f.path + ": " + why + " (6 attempts); run the command again to resume from " +
                  size_text(have));
    if (why.rfind("the server sent", 0) == 0) {  // not resumable: from the start
      prog.add(-have, f.path);
      if (::ftruncate(fd, 0) != 0) throw Error("cannot write " + part + ": " + std::strerror(errno));
      have = 0;
      dg.reset();
    }
    int wait = 1 << attempt;
    if (r.status == 429) wait = std::max(wait, std::min(60, std::atoi(r.header("retry-after").c_str())));
    std::fprintf(stderr, "\n[espetero] %s: %s; resuming at %s in %d s\n", f.path.c_str(), why.c_str(), size_text(have).c_str(),
                 wait);
    std::this_thread::sleep_for(std::chrono::seconds(wait));
  }
  if (const std::string got = dg.hex(); got != f.blob) {
    ::unlink(part.c_str());
    throw Error(info.repo + ": " + f.path + " does not match the Hub's checksum (" + (f.lfs ? "sha256 " : "git blob id ") +
                f.blob + ", got " + got + "); the partial file was removed, run the command again");
  }
  if (::rename(part.c_str(), blob.c_str()) != 0) throw Error("cannot rename " + part + ": " + std::strerror(errno));
}

// snapshots/<commit>/<path> -> ../../blobs/<blob> (one more ../ per directory of the path).
void link_blob(const std::string& repo, const std::string& commit, const RepoFile& f) {
  const std::string link = repo + "/snapshots/" + commit + "/" + f.path;
  mkdirs(dirname_of(link));
  std::string target = "../../blobs/" + f.blob;
  for (char c : f.path)
    if (c == '/') target = "../" + target;
  struct stat st;
  if (::lstat(link.c_str(), &st) == 0) ::unlink(link.c_str());
  if (::symlink(target.c_str(), link.c_str()) != 0)
    throw Error("cannot link " + link + " to " + target + ": " + std::strerror(errno));
}

}  // namespace

void download(const std::string& cache, const RepoInfo& info, const std::vector<std::string>& paths, const std::string& label,
              bool quiet) {
  const size_t slash = info.repo.find('/');
  const std::string folder = "models--" + info.repo.substr(0, slash) + "--" + info.repo.substr(slash + 1);
  const std::string repo = cache + "/" + folder, locks = cache + "/.locks/" + folder;
  for (const std::string& d : {repo + "/blobs", repo + "/refs", repo + "/snapshots/" + info.commit, locks}) mkdirs(d);
  // refs/<revision> (no newline), as huggingface_hub writes it before the
  // files: here once the disk has the room for them.
  auto write_ref = [&] {
    if (info.revision == info.commit || !safe_path(info.revision) || info.revision.find('/') != std::string::npos) return;
    const std::string ref = repo + "/refs/" + info.revision, tmp = ref + ".tmp";
    std::ofstream(tmp, std::ios::binary | std::ios::trunc) << info.commit;
    if (::rename(tmp.c_str(), ref.c_str()) != 0) throw Error("cannot write " + ref + ": " + std::strerror(errno));
  };
  struct Job {
    const RepoFile* f;
    int64_t have;  // bytes of an earlier attempt
  };
  std::vector<Job> jobs;
  int64_t total = 0, done = 0;
  for (const std::string& p : paths) {
    const RepoFile* f = info.find(p);
    if (!f) throw Error(info.repo + ": no file " + p + " at revision " + info.commit);
    if (is_file(repo + "/snapshots/" + info.commit + "/" + p)) continue;  // linked to a blob that is there
    if (is_file(repo + "/blobs/" + f->blob)) {
      link_blob(repo, info.commit, *f);  // another revision or file has the same content
      continue;
    }
    const int64_t have = std::min<int64_t>(f->size, std::max<int64_t>(0, size_of(repo + "/blobs/" + f->blob + ".incomplete")));
    jobs.push_back({f, have});
    total += f->size;
    done += have;
  }
  if (jobs.empty()) {
    write_ref();
    return;
  }
  // The disk, before anything is written (ESPETERO_HUB_DISK_FREE: a test's answer, in bytes).
  struct statvfs vs;
  int64_t free_bytes = ::statvfs((repo + "/blobs").c_str(), &vs) == 0 ? int64_t(vs.f_bavail) * int64_t(vs.f_frsize) : INT64_MAX;
  if (const char* t = std::getenv("ESPETERO_HUB_DISK_FREE"); t && *t) free_bytes = std::atoll(t);
  const int64_t need = total - done;
  if (need + (int64_t(256) << 20) > free_bytes)
    throw Error(label + " needs " + models::gb(need) + " on the disk of " + models::tilde(cache) + ", which has " +
                models::gb(free_bytes) + " free: its files were not downloaded. Free some space or put the cache on a larger "
                "disk with HF_HUB_CACHE");
  write_ref();
  if (!quiet)
    std::fprintf(stderr, "[espetero] downloading %s: %s in %zu file%s into %s\n", label.c_str(), size_text(need).c_str(),
                 jobs.size(), jobs.size() == 1 ? "" : "s", models::tilde(repo).c_str());
  Progress prog(total, done, quiet);
  std::atomic<size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex mu;
  std::string first;
  auto worker = [&] {
    for (size_t i; (i = next++) < jobs.size() && !stop;) {
      const RepoFile& f = *jobs[i].f;
      const std::string blob = repo + "/blobs/" + f.blob, lock = locks + "/" + f.blob + ".lock";
      try {
        const int lfd = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (lfd < 0) throw Error("cannot open " + lock + ": " + std::strerror(errno));
        struct Unlock {
          int fd;
          ~Unlock() { ::close(fd); }  // closing releases the flock
        } unlock{lfd};
        if (::flock(lfd, LOCK_EX | LOCK_NB) != 0) {
          std::fprintf(stderr, "\n[espetero] %s: another process is downloading it; waiting\n", f.path.c_str());
          while (::flock(lfd, LOCK_EX) != 0)
            if (errno != EINTR) throw Error("cannot lock " + lock + ": " + std::strerror(errno));
        }
        if (is_file(blob)) {  // the other process finished it
          prog.add(f.size - jobs[i].have, f.path);
        } else {
          fetch_blob(info, f, blob, prog, stop);
        }
        link_blob(repo, info.commit, f);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu);
        if (first.empty() && std::string(e.what()) != "stopped") first = e.what();
        stop = true;
      }
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 0; t < std::min<size_t>(4, jobs.size()); ++t) pool.emplace_back(worker);
  for (std::thread& t : pool) t.join();
  prog.finish(first.empty());
  if (!first.empty()) throw Error(first);
}

std::string read_range(const RepoInfo& info, const std::string& path, int64_t offset, int64_t length) {
  if (length <= 0) return "";
  const std::string range = "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + length - 1);
  const HttpResponse r = http_get(resolve_url(info.repo, info.commit, path), {{"Range", range}, {"Accept-Encoding", "identity"}},
                                  size_t(length), 120);
  if (!r.error.empty() || (r.status != 206 && r.status != 200)) throw Error(hub_error(info.repo, path, r));
  if (r.status == 200 && offset > 0) throw Error(info.repo + ": " + path + ": the server ignored a byte range request");
  return r.body;
}

}  // namespace espetero::hub
