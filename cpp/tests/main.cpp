/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

// class Environment : public ::testing::Environment {
//  public:
//   Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

//   void SetUp() override
//   {
//     const char* argv[] = {"./test", "--gpus", "2"};  // TODO: make configurable
//     int argc           = 3;
//     EXPECT_EQ(legate::start(argc, (char**)argv), 0);
//   }
//   void TearDown() override { EXPECT_EQ(legate::finish(), 0); }

//  private:
//   int argc_;
//   char** argv_;
// };

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  // ::testing::AddGlobalTestEnvironment(new Environment(argc, argv));
  return RUN_ALL_TESTS();
}
