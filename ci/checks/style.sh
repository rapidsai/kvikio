#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#######################
# kvikio Style Tester #
#######################

PATH=/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
