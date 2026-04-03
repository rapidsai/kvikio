/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <curl/curl.h>

#include <kvikio/aws_credential_provider.hpp>
#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio {

namespace {

constexpr int kImdsConnectTimeoutSecs = 2;
constexpr int kImdsTotalTimeoutSecs   = 5;
constexpr int kImdsTokenTtlSecs       = 21'600;

using clock = std::chrono::system_clock;

std::string trim_in_place(std::string s)
{
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

std::optional<std::string> getenv_nonempty(char const* name)
{
  char const* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') { return std::nullopt; }
  return std::string{v};
}

std::string imds_base_url(std::optional<std::string> override_url)
{
  if (override_url.has_value() && !override_url->empty()) {
    std::string b = *override_url;
    while (!b.empty() && b.back() == '/') {
      b.pop_back();
    }
    return b;
  }
  if (auto e = getenv_nonempty("AWS_EC2_METADATA_SERVICE_ENDPOINT")) { return *e; }
  return std::string{"http://169.254.169.254"};
}

std::string join_path(std::string const& base, std::string_view rel)
{
  std::string out = base;
  if (!out.empty() && out.back() == '/') { out.pop_back(); }
  out.push_back('/');
  out.append(rel);
  return out;
}

clock::time_point parse_aws_expiration_iso8601(std::string const& iso)
{
  std::tm tm{};
  std::istringstream ss(iso);
  ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  if (ss.fail()) {
    KVIKIO_FAIL("IMDS: cannot parse credential Expiration: " + iso, std::runtime_error);
  }
#if defined(_WIN32)
  std::time_t tt = _mkgmtime(&tm);
#else
  std::time_t tt = timegm(&tm);
#endif
  if (tt == static_cast<std::time_t>(-1)) {
    KVIKIO_FAIL("IMDS: timegm failed for Expiration: " + iso, std::runtime_error);
  }
  return clock::from_time_t(tt);
}

std::optional<std::string> json_extract_string(std::string_view json, std::string_view key)
{
  std::string needle = "\"";
  needle.append(key);
  needle.push_back('"');
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) { return std::nullopt; }
  pos += needle.size();
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != ':') { return std::nullopt; }
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != '"') { return std::nullopt; }
  ++pos;
  std::string out;
  for (; pos < json.size(); ++pos) {
    char c = json[pos];
    if (c == '"') { return out; }
    if (c == '\\' && pos + 1 < json.size()) {
      ++pos;
      out.push_back(json[pos]);
      continue;
    }
    out.push_back(c);
  }
  return std::nullopt;
}

void imds_apply_timeouts(CurlHandle& curl)
{
  curl.setopt(CURLOPT_CONNECTTIMEOUT, static_cast<long>(kImdsConnectTimeoutSecs));
  curl.setopt(CURLOPT_TIMEOUT, static_cast<long>(kImdsTotalTimeoutSecs));
}

size_t empty_upload_read(char*, size_t, size_t, void*) { return 0; }

void imds_put_token(CurlHandle& curl, std::string const& token_url, std::string& token_out)
{
  token_out.clear();
  std::string ttl = std::to_string(kImdsTokenTtlSecs);
  curl_slist* hdrs =
    curl_slist_append(nullptr, ("X-aws-ec2-metadata-token-ttl-seconds: " + ttl).c_str());
  if (hdrs == nullptr) { KVIKIO_FAIL("IMDS: curl_slist_append failed", std::runtime_error); }
  curl.setopt(CURLOPT_URL, token_url.c_str());
  curl.setopt(CURLOPT_CUSTOMREQUEST, "PUT");
  curl.setopt(CURLOPT_UPLOAD, 1L);
  curl.setopt(CURLOPT_READFUNCTION, empty_upload_read);
  curl.setopt(CURLOPT_READDATA, nullptr);
  curl.setopt(CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(0));
  curl.setopt(CURLOPT_HTTPHEADER, hdrs);
  curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_get_string_response);
  curl.setopt(CURLOPT_WRITEDATA, &token_out);
  imds_apply_timeouts(curl);
  curl.perform();
  curl_slist_free_all(hdrs);
  curl.setopt(CURLOPT_UPLOAD, 0L);
  curl.setopt(CURLOPT_READFUNCTION, nullptr);
  curl.setopt(CURLOPT_READDATA, nullptr);
  curl.setopt(CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(0));
  curl.setopt(CURLOPT_CUSTOMREQUEST, nullptr);
  curl.setopt(CURLOPT_HTTPHEADER, nullptr);
  token_out = trim_in_place(std::move(token_out));
  if (token_out.empty()) {
    KVIKIO_FAIL("IMDS: empty metadata session token from " + token_url, std::runtime_error);
  }
}

void imds_get_with_token(CurlHandle& curl,
                         std::string const& url,
                         std::string const& metadata_token,
                         std::string& body_out)
{
  body_out.clear();
  std::string hdr_line = "X-aws-ec2-metadata-token: " + metadata_token;
  curl_slist* hdrs     = curl_slist_append(nullptr, hdr_line.c_str());
  if (hdrs == nullptr) { KVIKIO_FAIL("IMDS: curl_slist_append failed", std::runtime_error); }
  curl.setopt(CURLOPT_HTTPGET, 1L);
  curl.setopt(CURLOPT_URL, url.c_str());
  curl.setopt(CURLOPT_HTTPHEADER, hdrs);
  curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_get_string_response);
  curl.setopt(CURLOPT_WRITEDATA, &body_out);
  imds_apply_timeouts(curl);
  curl.perform();
  curl_slist_free_all(hdrs);
  curl.setopt(CURLOPT_HTTPHEADER, nullptr);
}

void fetch_imds_credentials(std::string const& base_url,
                            std::string& access_key_out,
                            std::string& secret_out,
                            std::string& session_token_out,
                            clock::time_point& expiration_out)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto curl = create_curl_handle();
  std::string meta_token;
  imds_put_token(curl, join_path(base_url, "latest/api/token"), meta_token);

  std::string role_name_raw;
  imds_get_with_token(curl,
                      join_path(base_url, "latest/meta-data/iam/security-credentials/"),
                      meta_token,
                      role_name_raw);
  auto role_name = trim_in_place(std::move(role_name_raw));
  if (role_name.empty()) {
    KVIKIO_FAIL(
      "IMDS: no IAM role name at latest/meta-data/iam/security-credentials/ "
      "(is an instance role attached to this host?)",
      std::runtime_error);
  }

  std::string cred_json;
  imds_get_with_token(curl,
                      join_path(base_url, "latest/meta-data/iam/security-credentials/" + role_name),
                      meta_token,
                      cred_json);

  auto access = json_extract_string(cred_json, "AccessKeyId");
  auto secret = json_extract_string(cred_json, "SecretAccessKey");
  auto token  = json_extract_string(cred_json, "Token");
  auto exp    = json_extract_string(cred_json, "Expiration");
  if (!access.has_value() || !secret.has_value() || !token.has_value() || !exp.has_value()) {
    KVIKIO_FAIL("IMDS: missing fields in security-credentials response", std::runtime_error);
  }
  access_key_out    = std::move(*access);
  secret_out        = std::move(*secret);
  session_token_out = std::move(*token);
  expiration_out    = parse_aws_expiration_iso8601(*exp);
}

class StaticCredentialProvider : public AwsCredentialProvider {
  std::mutex mutex_;
  std::string access_;
  std::string secret_;
  std::optional<std::string> session_;
  std::shared_ptr<AwsAuthMaterial const> cache_;

 public:
  StaticCredentialProvider(std::string access,
                           std::string secret,
                           std::optional<std::string> session)
    : access_{std::move(access)}, secret_{std::move(secret)}, session_{std::move(session)}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_) { return cache_; }
    if (access_.compare(0, 4, "ASIA") == 0) {
      KVIKIO_EXPECT(session_.has_value() && !session_->empty(),
                    "Static AWS credentials: session token required when access key id begins "
                    "with ASIA",
                    std::invalid_argument);
    }
    cache_ = AwsAuthMaterial::create(access_, secret_, session_);
    return cache_;
  }
};

class EnvironmentCredentialProvider : public AwsCredentialProvider {
  std::mutex mutex_;
  std::shared_ptr<AwsAuthMaterial const> cache_;

 public:
  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_) { return cache_; }
    auto access                        = detail::unwrap_or_env(std::nullopt,
                                        "AWS_ACCESS_KEY_ID",
                                        "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID "
                                                               "isn't set.");
    auto secret                        = detail::unwrap_or_env(std::nullopt,
                                        "AWS_SECRET_ACCESS_KEY",
                                        "S3: must provide `aws_secret_access_key` if "
                                                               "AWS_SECRET_ACCESS_KEY isn't set.");
    std::optional<std::string> session = std::nullopt;
    if (access->compare(0, 4, std::string("ASIA")) == 0) {
      session = detail::unwrap_or_env(std::nullopt,
                                      "AWS_SESSION_TOKEN",
                                      "When using temporary credentials, AWS_SESSION_TOKEN must "
                                      "be set.");
    }
    cache_ = AwsAuthMaterial::create(*access, *secret, session);
    return cache_;
  }
};

class LegacyEnvAndArgsCredentialProvider : public AwsCredentialProvider {
  std::mutex mutex_;
  std::optional<std::string> opt_access_;
  std::optional<std::string> opt_secret_;
  std::optional<std::string> opt_session_;
  std::shared_ptr<AwsAuthMaterial const> cache_;

 public:
  LegacyEnvAndArgsCredentialProvider(std::optional<std::string> aws_access_key,
                                     std::optional<std::string> aws_secret_access_key,
                                     std::optional<std::string> aws_session_token)
    : opt_access_{std::move(aws_access_key)},
      opt_secret_{std::move(aws_secret_access_key)},
      opt_session_{std::move(aws_session_token)}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_) { return cache_; }
    auto access                        = detail::unwrap_or_env(std::move(opt_access_),
                                        "AWS_ACCESS_KEY_ID",
                                        "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID "
                                                               "isn't set.");
    auto secret                        = detail::unwrap_or_env(std::move(opt_secret_),
                                        "AWS_SECRET_ACCESS_KEY",
                                        "S3: must provide `aws_secret_access_key` if "
                                                               "AWS_SECRET_ACCESS_KEY isn't set.");
    std::optional<std::string> session = std::nullopt;
    if (access->compare(0, 4, std::string("ASIA")) == 0) {
      session = detail::unwrap_or_env(std::move(opt_session_),
                                      "AWS_SESSION_TOKEN",
                                      "When using temporary credentials, AWS_SESSION_TOKEN must "
                                      "be set.");
    }
    cache_ = AwsAuthMaterial::create(*access, *secret, session);
    return cache_;
  }
};

class IamRoleCredentialProvider : public AwsCredentialProvider {
  std::mutex mutex_;
  std::string base_url_;
  std::shared_ptr<AwsAuthMaterial const> material_;
  clock::time_point refresh_after_{};

 public:
  explicit IamRoleCredentialProvider(std::optional<std::string> endpoint_override)
    : base_url_{imds_base_url(std::move(endpoint_override))}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const now = clock::now();
    if (material_ && now < refresh_after_) { return material_; }

    std::string ak;
    std::string sk;
    std::string tok;
    clock::time_point expiration{};
    fetch_imds_credentials(base_url_, ak, sk, tok, expiration);

    constexpr auto skew = std::chrono::minutes{5};
    auto next_refresh   = expiration - skew;
    if (next_refresh <= now) { next_refresh = now + std::chrono::seconds{30}; }
    refresh_after_ = next_refresh;
    material_      = AwsAuthMaterial::create(std::move(ak), std::move(sk), std::move(tok));
    return material_;
  }
};

class DefaultCredentialProvider : public AwsCredentialProvider {
  std::mutex mutex_;
  std::optional<std::string> imds_override_;
  std::shared_ptr<AwsCredentialProvider> inner_;

 public:
  explicit DefaultCredentialProvider(std::optional<std::string> imds_override)
    : imds_override_{std::move(imds_override)}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inner_) {
      auto id  = getenv_nonempty("AWS_ACCESS_KEY_ID");
      auto key = getenv_nonempty("AWS_SECRET_ACCESS_KEY");
      if (id.has_value() && key.has_value()) {
        inner_ = std::make_shared<EnvironmentCredentialProvider>();
      } else {
        inner_ = std::make_shared<IamRoleCredentialProvider>(imds_override_);
      }
    }
    return inner_->get_auth_material();
  }
};

}  // namespace

AwsAuthMaterial::AwsAuthMaterial() = default;

AwsAuthMaterial::~AwsAuthMaterial()
{
  curl_slist_free_all(token_header_list);
  token_header_list = nullptr;
}

std::shared_ptr<AwsAuthMaterial const> AwsAuthMaterial::create(
  std::string access_key_id,
  std::string secret_access_key,
  std::optional<std::string> session_token)
{
  bool const is_asia = access_key_id.size() >= 4 && access_key_id.compare(0, 4, "ASIA") == 0;
  auto m             = std::shared_ptr<AwsAuthMaterial>(new AwsAuthMaterial());
  m->userpwd         = std::move(access_key_id);
  m->userpwd.push_back(':');
  m->userpwd.append(secret_access_key);

  if (session_token.has_value() && !session_token->empty()) {
    std::string line = "x-amz-security-token: ";
    line += *session_token;
    m->token_header_list = curl_slist_append(nullptr, line.c_str());
    KVIKIO_EXPECT(m->token_header_list != nullptr,
                  "Failed to create curl header for AWS token",
                  std::runtime_error);
  } else {
    KVIKIO_EXPECT(!is_asia,
                  "AWS session token required for temporary access key ids (ASIA...)",
                  std::invalid_argument);
  }
  return m;
}

std::shared_ptr<AwsCredentialProvider> make_legacy_env_and_args_credential_provider(
  std::optional<std::string> aws_access_key,
  std::optional<std::string> aws_secret_access_key,
  std::optional<std::string> aws_session_token)
{
  return std::make_shared<LegacyEnvAndArgsCredentialProvider>(
    std::move(aws_access_key), std::move(aws_secret_access_key), std::move(aws_session_token));
}

std::shared_ptr<AwsCredentialProvider> make_default_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override)
{
  return std::make_shared<DefaultCredentialProvider>(std::move(imds_endpoint_override));
}

std::shared_ptr<AwsCredentialProvider> make_aws_credential_provider(
  AwsCredentialKind kind,
  std::optional<std::string> aws_access_key,
  std::optional<std::string> aws_secret_access_key,
  std::optional<std::string> aws_session_token,
  std::optional<std::string> imds_endpoint_override)
{
  switch (kind) {
    case AwsCredentialKind::Default:
      return make_default_aws_credential_provider(std::move(imds_endpoint_override));
    case AwsCredentialKind::Environment: return std::make_shared<EnvironmentCredentialProvider>();
    case AwsCredentialKind::Static:
      KVIKIO_EXPECT(aws_access_key.has_value() && !aws_access_key->empty(),
                    "Static AWS credentials require a non-empty access key id",
                    std::invalid_argument);
      KVIKIO_EXPECT(aws_secret_access_key.has_value() && !aws_secret_access_key->empty(),
                    "Static AWS credentials require a non-empty secret access key",
                    std::invalid_argument);
      return std::make_shared<StaticCredentialProvider>(std::move(*aws_access_key),
                                                        std::move(*aws_secret_access_key),
                                                        std::move(aws_session_token));
    case AwsCredentialKind::IamRole:
      return std::make_shared<IamRoleCredentialProvider>(std::move(imds_endpoint_override));
    case AwsCredentialKind::Legacy:
      return make_legacy_env_and_args_credential_provider(
        std::move(aws_access_key), std::move(aws_secret_access_key), std::move(aws_session_token));
    default: KVIKIO_FAIL("Unknown AwsCredentialKind", std::invalid_argument);
  }
}

}  // namespace kvikio
