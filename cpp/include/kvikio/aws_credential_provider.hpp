/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef KVIKIO_LIBCURL_FOUND
#error \
  "cannot include the remote IO API, please build KvikIO with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
#endif

#include <memory>
#include <optional>
#include <string>

struct curl_slist;

namespace kvikio {

/**
 * @brief Immutable AWS SigV4 user/password and optional session-token header for libcurl.
 *
 * `token_header_list` must outlive `curl_easy_perform`; callers hold `shared_ptr` to this object
 * until the transfer completes.
 */
class AwsAuthMaterial {
 public:
  std::string userpwd;
  ::curl_slist* token_header_list{};

  AwsAuthMaterial();
  ~AwsAuthMaterial();
  AwsAuthMaterial(AwsAuthMaterial const&)            = delete;
  AwsAuthMaterial& operator=(AwsAuthMaterial const&) = delete;
  AwsAuthMaterial(AwsAuthMaterial&&)                 = delete;
  AwsAuthMaterial& operator=(AwsAuthMaterial&&)      = delete;

  static std::shared_ptr<AwsAuthMaterial const> create(std::string access_key_id,
                                                       std::string secret_access_key,
                                                       std::optional<std::string> session_token);
};

/**
 * @brief How Python / Cython select the AWS credential source for S3.
 */
enum class AwsCredentialKind : std::uint8_t {
  Default     = 0,  ///< Environment keys if set, else IAM role via metadata (IMDSv2)
  Environment = 1,  ///< `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / optional token only
  Static      = 2,  ///< Explicit access key, secret, optional session token
  IamRole     = 3,  ///< IAM role credentials from the compute metadata service (IMDSv2) only
  Legacy      = 4,  ///< Optional args plus environment (pre-credential S3 API semantics)
};

class AwsCredentialProvider {
 public:
  virtual ~AwsCredentialProvider() = default;

  /**
   * @brief Return auth material for one HTTP request; implementations cache and refresh as needed.
   */
  virtual std::shared_ptr<AwsAuthMaterial const> get_auth_material() = 0;
};

/**
 * @brief Build a credential provider for the given kind (used by Cython).
 *
 * @param kind Credential selection mode
 * @param aws_access_key Required when kind == Static; optional when kind == Legacy (env fallback)
 * @param aws_secret_access_key Required when kind == Static
 * @param aws_session_token Optional; required when access key begins with "ASIA" (Static / Legacy)
 * @param imds_endpoint_override Optional base URL (e.g. http://127.0.0.1:1234) for tests; if
 *        nullopt, uses `AWS_EC2_METADATA_SERVICE_ENDPOINT` or the default EC2 link-local address.
 */
std::shared_ptr<AwsCredentialProvider> make_aws_credential_provider(
  AwsCredentialKind kind,
  std::optional<std::string> aws_access_key         = std::nullopt,
  std::optional<std::string> aws_secret_access_key  = std::nullopt,
  std::optional<std::string> aws_session_token      = std::nullopt,
  std::optional<std::string> imds_endpoint_override = std::nullopt);

/**
 * @brief Provider matching legacy S3Endpoint optional arguments plus environment variables.
 */
std::shared_ptr<AwsCredentialProvider> make_legacy_env_and_args_credential_provider(
  std::optional<std::string> aws_access_key,
  std::optional<std::string> aws_secret_access_key,
  std::optional<std::string> aws_session_token);

/**
 * @brief Default chain: static env keys if both `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
 * are set and non-empty, otherwise IAM role credentials via the metadata service (IMDSv2).
 */
std::shared_ptr<AwsCredentialProvider> make_default_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override = std::nullopt);

}  // namespace kvikio
