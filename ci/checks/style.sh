#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
#######################
# kvikio Style Tester #
#######################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run formatting script
FORMAT=`python scripts/format-all.py --check 2>&1`
FORMAT_RETVAL=$?

if [ "$FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: format check; begin output\n\n"
  echo -e "$FORMAT"
  echo -e "\n\n>>>> FAILED: format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: format check\n\n"
fi

RETVALS=(
  $FORMAT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL

