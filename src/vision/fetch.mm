// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Images by URL: the fetch, with Foundation's NSURLSession (one
// ephemeral session per image: no cookies, cache, credentials or proxies),
// under the limits of fetch.h. The session's delegate checks the response
// (2xx, image/*, size), every redirect (count, scheme, the new host's
// addresses) and the address each connection used (metrics).
#import <Foundation/Foundation.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "../server/version.h"
#include "fetch.h"

namespace {

using Clock = std::chrono::steady_clock;

// Shared by the caller and the session's delegate queue (which may still
// deliver a late callback after the caller has returned).
struct FetchState {
  espetero::vision::FetchOptions opt;
  Clock::time_point deadline;
  std::mutex mu;
  std::condition_variable cv;
  std::string bytes, mime, err;
  int redirects = 0;
  bool done = false, metrics = false;
  double left() const { return std::chrono::duration<double>(deadline - Clock::now()).count(); }
  void fail(const std::string& e) {  // the first reason wins
    std::lock_guard<std::mutex> lk(mu);
    if (err.empty()) err = e;
  }
};

std::string str(NSString* s) { return s ? std::string([s UTF8String] ?: "") : std::string(); }

// http or https, a host, no user name or password; `host` gets the host.
bool url_ok(NSURL* url, std::string& host, std::string& err) {
  if (!url) {
    err = "not a valid URL";
    return false;
  }
  NSURLComponents* c = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
  const std::string scheme = str(c.scheme.lowercaseString);
  if (scheme != "http" && scheme != "https") {
    err = "only http and https URLs are fetched";
    return false;
  }
  if (c.user.length || c.password.length) {
    err = "URLs with a user name or password are not fetched";
    return false;
  }
  host = str(url.host);  // as the connection uses it: percent-decoded, IDNA-encoded, no IPv6 brackets
  if (host.empty()) {
    err = "the URL has no host";
    return false;
  }
  return true;
}

}  // namespace

@interface EspeteroImageFetch : NSObject <NSURLSessionDataDelegate>
@end

@implementation EspeteroImageFetch {
 @public
  std::shared_ptr<FetchState> st;
}

- (void)URLSession:(NSURLSession*)session
              dataTask:(NSURLSessionDataTask*)task
    didReceiveResponse:(NSURLResponse*)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))handler {
  FetchState& s = *st;
  std::string why;
  NSHTTPURLResponse* http = [response isKindOfClass:[NSHTTPURLResponse class]] ? (NSHTTPURLResponse*)response : nil;
  const std::string mime = str(response.MIMEType.lowercaseString);
  if (!http) why = "not an HTTP response";
  else if (http.statusCode < 200 || http.statusCode > 299) why = "HTTP status " + std::to_string(long(http.statusCode));
  else if (mime.compare(0, 6, "image/") != 0) why = "content type '" + (mime.empty() ? std::string("none") : mime) + "' is not image/*";
  else if (response.expectedContentLength > 0 && uint64_t(response.expectedContentLength) > s.opt.max_bytes)
    why = "the image is " + std::to_string(static_cast<long long>(response.expectedContentLength)) + " bytes, over the limit of " +
          std::to_string(s.opt.max_bytes);
  if (!why.empty()) {
    s.fail(why);
    handler(NSURLSessionResponseCancel);
    return;
  }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.mime = mime;
  }
  handler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task didReceiveData:(NSData*)data {
  FetchState& s = *st;
  bool over = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.bytes.size() + data.length > s.opt.max_bytes) over = true;
    else s.bytes.append(static_cast<const char*>(data.bytes), data.length);
  }
  if (over) {
    s.fail("the image is over the limit of " + std::to_string(s.opt.max_bytes) + " bytes");
    [task cancel];
  }
}

- (void)URLSession:(NSURLSession*)session
                          task:(NSURLSessionTask*)task
    willPerformHTTPRedirection:(NSHTTPURLResponse*)response
                    newRequest:(NSURLRequest*)request
             completionHandler:(void (^)(NSURLRequest*))handler {
  FetchState& s = *st;
  std::string host, why;
  if (++s.redirects > s.opt.max_redirects) why = "more than " + std::to_string(s.opt.max_redirects) + " redirects";
  else if (!url_ok(request.URL, host, why)) why = "redirected to " + str(request.URL.absoluteString).substr(0, 200) + ": " + why;
  else if (!espetero::vision::host_allowed(host, s.opt, std::max(0.1, s.left()), why)) why = "redirected: " + why;
  if (!why.empty()) {
    s.fail(why);
    handler(nil);
    [task cancel];
    return;
  }
  handler(request);
}

- (void)URLSession:(NSURLSession*)session
                   task:(NSURLSessionTask*)task
    didReceiveChallenge:(NSURLAuthenticationChallenge*)challenge
      completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition, NSURLCredential*))handler {
  // Server trust is checked as usual; no credentials are ever sent.
  if ([challenge.protectionSpace.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust])
    handler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
  else
    handler(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task didFinishCollectingMetrics:(NSURLSessionTaskMetrics*)metrics {
  FetchState& s = *st;
  // The addresses really connected to: a name that resolved to a public
  // address for the check and to a private one for the connection (DNS
  // rebinding) gets no image back. This reason replaces any earlier one, so
  // the status, type or size that an internal service answered is never
  // reported to the client. A transaction with a response but no address
  // (unexpected: the address is missing only when no connection was made)
  // cannot be vouched for and fails too.
  if (!s.opt.allow_private)
    for (NSURLSessionTaskTransactionMetrics* t in metrics.transactionMetrics) {
      const std::string addr = str(t.remoteAddress);
      std::string why;
      if (addr.empty()) {
        if (t.response) why = "the connection's address could not be checked";
      } else if (espetero::vision::blocked_address_text(addr, s.opt.blocked)) {
        why = "connected to " + addr + ", a loopback, private, link-local or otherwise non-public address (the operator "
              "can allow those with --image-urls-private)";
      }
      if (!why.empty()) {
        std::lock_guard<std::mutex> lk(s.mu);
        s.err = why;
        break;
      }
    }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.metrics = true;
  }
  s.cv.notify_all();
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task didCompleteWithError:(NSError*)error {
  FetchState& s = *st;
  if (error) {
    if (error.code == NSURLErrorTimedOut) s.fail("timed out after " + std::to_string(int(s.opt.timeout_s)) + " s");
    else if (error.code != NSURLErrorCancelled) s.fail(str(error.localizedDescription));
    else s.fail("cancelled");
  }
  {
    std::lock_guard<std::mutex> lk(s.mu);
    s.done = true;
  }
  s.cv.notify_all();
}

@end

namespace espetero::vision {

bool fetch_image(const std::string& url_text, const FetchOptions& opt, std::string& bytes, std::string& mime,
                 std::string& err) {
  @autoreleasepool {
    auto st = std::make_shared<FetchState>();
    st->opt = opt;
    st->deadline = Clock::now() + std::chrono::milliseconds(int64_t(opt.timeout_s * 1000));
    NSString* text = [NSString stringWithUTF8String:url_text.c_str()];
    NSURL* url = text ? [NSURL URLWithString:text] : nil;
    std::string host;
    if (!url_ok(url, host, err)) return false;
    if (!host_allowed(host, opt, st->left(), err)) return false;
    NSURLSessionConfiguration* cfg = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    const double left = std::max(0.5, st->left());
    cfg.timeoutIntervalForRequest = left;
    cfg.timeoutIntervalForResource = left;
    cfg.HTTPCookieStorage = nil;
    cfg.HTTPShouldSetCookies = NO;
    cfg.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
    cfg.URLCache = nil;
    cfg.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
    cfg.URLCredentialStorage = nil;
    cfg.connectionProxyDictionary = @{};  // no proxy: the address checked is the one connected to
    cfg.waitsForConnectivity = NO;
    cfg.HTTPMaximumConnectionsPerHost = 1;
    cfg.HTTPAdditionalHeaders = @{
      @"User-Agent" : [NSString stringWithFormat:@"%s/%s", espetero::kProduct, espetero::kVersion],
      @"Accept" : @"image/*"
    };
    EspeteroImageFetch* d = [[EspeteroImageFetch alloc] init];
    d->st = st;
    NSOperationQueue* q = [[NSOperationQueue alloc] init];
    q.maxConcurrentOperationCount = 1;
    NSURLSession* session = [NSURLSession sessionWithConfiguration:cfg delegate:d delegateQueue:q];
    NSURLSessionDataTask* task = [session dataTaskWithURL:url];
    [task resume];
    {
      std::unique_lock<std::mutex> lk(st->mu);
      // The session's own timeout ends the task; a second later we stop waiting regardless.
      if (!st->cv.wait_until(lk, st->deadline + std::chrono::seconds(1), [&] { return st->done; })) {
        lk.unlock();
        st->fail("timed out after " + std::to_string(int(opt.timeout_s)) + " s");
        [task cancel];
        lk.lock();
        st->cv.wait_for(lk, std::chrono::seconds(2), [&] { return st->done; });
      }
      // Metrics come with the task's end; wait briefly for them (they carry the address check).
      if (!opt.allow_private) st->cv.wait_for(lk, std::chrono::seconds(1), [&] { return st->metrics; });
    }
    [session invalidateAndCancel];  // releases the delegate
    std::lock_guard<std::mutex> lk(st->mu);
    if (!st->err.empty()) {
      err = st->err;
      return false;
    }
    if (!opt.allow_private && !st->metrics) {
      err = "the connection's address could not be checked";
      return false;
    }
    if (st->bytes.empty()) {
      err = "empty response";
      return false;
    }
    bytes = std::move(st->bytes);
    mime = st->mime;
    return true;
  }
}

bool fetch_image_data_url(const std::string& url, const FetchOptions& opt, std::string& data_url, std::string& err) {
  std::string bytes, mime;
  if (!fetch_image(url, opt, bytes, mime, err)) return false;
  @autoreleasepool {
    NSData* d = [NSData dataWithBytesNoCopy:bytes.data() length:bytes.size() freeWhenDone:NO];
    NSString* b64 = [d base64EncodedStringWithOptions:0];
    data_url = "data:" + mime + ";base64," + str(b64);
  }
  return true;
}

}  // namespace espetero::vision
