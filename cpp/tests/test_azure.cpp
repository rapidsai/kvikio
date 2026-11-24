/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <stdexcept>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/azure.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

using kvikio::AzureBlobSharedKeyEndpoint;
using kvikio::RemoteEndpointType;
using kvikio::RemoteHandle;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::ThrowsMessage;

namespace {

std::string const blob_url{"https://myaccount.blob.core.windows.net/mycontainer/myblob"};

// A made-up account key. These tests send no request. Therefore any valid base64 text works.
std::string const account_key{"Zm9v"};

}  // namespace

TEST(AzureBlobSharedKeyEndpointTest, is_url_valid_accepts_blob_urls)
{
  for (std::string const url : {
         blob_url,
         // A blob name can contain `/`.
         std::string{"https://myaccount.blob.core.windows.net/mycontainer/mydir/myblob"},
       }) {
    EXPECT_TRUE(AzureBlobSharedKeyEndpoint::is_url_valid(url)) << url;
  }
}

TEST(AzureBlobSharedKeyEndpointTest, is_url_valid_rejects_other_urls)
{
  for (std::string const url : {
         // The scheme is not `https`.
         "http://myaccount.blob.core.windows.net/mycontainer/myblob",
         // The URL has a query.
         "https://myaccount.blob.core.windows.net/mycontainer/myblob?comp=list",
         // The host is not `<account>.blob.core.windows.net`.
         "https://example.com/mycontainer/myblob",
         // The account name has fewer than 3 characters.
         "https://ab.blob.core.windows.net/mycontainer/myblob",
         // The account name has a character other than a lowercase letter or a digit.
         "https://my-account.blob.core.windows.net/mycontainer/myblob",
         // The path has no blob name.
         "https://myaccount.blob.core.windows.net/mycontainer",
         "https://myaccount.blob.core.windows.net/mycontainer/",
       }) {
    EXPECT_FALSE(AzureBlobSharedKeyEndpoint::is_url_valid(url)) << url;
  }
}

TEST(AzureBlobSharedKeyEndpointTest, constructor_rejects_invalid_arguments)
{
  // The URL has no blob name.
  EXPECT_THROW(
    AzureBlobSharedKeyEndpoint("https://myaccount.blob.core.windows.net/mycontainer", account_key),
    std::invalid_argument);
  // The account key is empty.
  EXPECT_THROW(AzureBlobSharedKeyEndpoint(blob_url, ""), std::invalid_argument);
  // The account key is not valid base64.
  EXPECT_THROW(AzureBlobSharedKeyEndpoint(blob_url, "Zm9vYmF"), std::invalid_argument);
}

TEST(AzureBlobSharedKeyEndpointTest, constructor_rejects_different_keys_in_env_vars)
{
  std::string const other_account_key{"QUJD"};
  kvikio::test::EnvVarContext env_var_ctx{{"AZURE_STORAGE_ACCOUNT_KEY", account_key},
                                          {"AZURE_STORAGE_ACCESS_KEY", account_key},
                                          {"AZURE_STORAGE_KEY", other_account_key}};

  // The error message names the environment variables but must not show either key.
  EXPECT_THAT([] { AzureBlobSharedKeyEndpoint{blob_url}; },
              ThrowsMessage<std::invalid_argument>(AllOf(HasSubstr("with a different value"),
                                                         Not(HasSubstr(account_key)),
                                                         Not(HasSubstr(other_account_key)))));
}

TEST(AzureBlobSharedKeyEndpointTest, open_with_azure_type)
{
  // `RemoteHandle::open()` creates the endpoint without an account key. Therefore the endpoint
  // reads the key from the environment. All three names get the same key. Otherwise a different
  // key that is already set under another name would make the endpoint throw.
  kvikio::test::EnvVarContext env_var_ctx{{"AZURE_STORAGE_ACCOUNT_KEY", account_key},
                                          {"AZURE_STORAGE_ACCESS_KEY", account_key},
                                          {"AZURE_STORAGE_KEY", account_key}};

  // The file size, 1, is given. Therefore `RemoteHandle::open()` does not ask the server for it.
  auto const handle =
    RemoteHandle::open(blob_url, RemoteEndpointType::AZURE_BLOB_SHARED_KEY, std::nullopt, 1);
  EXPECT_EQ(handle.remote_endpoint_type(), RemoteEndpointType::AZURE_BLOB_SHARED_KEY);
}

TEST(AzureBlobSharedKeyEndpointTest, open_with_auto_type_picks_http)
{
  // AUTO mode tries HTTP before Azure, and the HTTP endpoint accepts every Azure Blob URL.
  auto const handle = RemoteHandle::open(blob_url, RemoteEndpointType::AUTO, std::nullopt, 1);
  EXPECT_EQ(handle.remote_endpoint_type(), RemoteEndpointType::HTTP);
}
