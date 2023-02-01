#!/bin/bash
##########################
# kvikIO Version Updater #
##########################

## Usage
# bash update-version.sh <new_version>

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG}).*"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# cpp update
sed_runner "/project(/,/)/s/VERSION.*/VERSION ${NEXT_FULL_TAG}/" cpp/CMakeLists.txt

# python update
sed_runner 's/set(kvikio_version.*)/set(kvikio_version '${NEXT_FULL_TAG}')/g' python/CMakeLists.txt

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' cpp/cmake/fetch_rapids.cmake

# script update
sed_runner 's/version=.*/version="'${NEXT_SHORT_TAG}'"):/g' scripts/format-all.py

# doxyfile update
sed_runner 's/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = '${NEXT_FULL_TAG}'/g' cpp/doxygen/Doxyfile

# sphinx docs update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

# bump cudf
for FILE in conda/environments/*.yaml dependencies.yaml; do
  sed_runner "s/cudf=.*/cudf=${NEXT_SHORT_TAG}/g" ${FILE};
done