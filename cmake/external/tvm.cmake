# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

SET(TVM_PREFIX_DIR ${THIRD_PARTY_PATH}/tvm)
SET(TVM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/tvm)
SET(TVM_INCLUDE_DIR "${TVM_INSTALL_DIR}/include" CACHE PATH "tvm include directory." FORCE)
SET(TVM_LIBRARIES "{TVM_INSTALL_DIR}/lib" CACHE FILEPATH "tvm library." FORCE)
SET(TVM_REPOSITORY https://github.com/apache/incubator-tvm.git)
SET(TVM_TAG master)

INCLUDE_DIRECTORIES(${TVM_INCLUDE_DIR})

cache_third_party(extern_tvm
    REPOSITORY   ${TVM_REPOSITORY}
    TAG          ${TVM_TAG})

ExternalProject_Add(
    extern_tvm
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${TVM_DOWNLOAD_CMD}"
    PREFIX ${TVM_PREFIX_DIR}
    CONFIGURE_COMMAND "" 
    BUILD_COMMAND mkdir -p ${TVM_PREFIX_DIR}/src/extern_tvm/build
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/cmake/config.cmake ${TVM_PREFIX_DIR}/src/extern_tvm/build
        && sed -i "s/USE_CUDA\ OFF/USE_CUDA\ ON/g" ${TVM_PREFIX_DIR}/src/extern_tvm/build/config.cmake
        && sed -i "s%USE_LLVM\ OFF%USE_LLVM\ /usr/lib/llvm-9.0/bin/llvm-config%g" ${TVM_PREFIX_DIR}/src/extern_tvm/build/config.cmake
        && cd ${TVM_PREFIX_DIR}/src/extern_tvm/build/ && cmake ..
        && make -j12
    INSTALL_COMMAND mkdir -p ${TVM_INSTALL_DIR}/include
        && mkdir -p ${TVM_INSTALL_DIR}/lib
        && cp -r ${TVM_PREFIX_DIR}/src/extern_tvm/include ${TVM_INSTALL_DIR}/
        && cp -r ${TVM_PREFIX_DIR}/src/extern_tvm/3rdparty/dmlc-core/include/dmlc ${TVM_INSTALL_DIR}/include/
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libtvm_runtime.so ${TVM_INSTALL_DIR}/lib
        #&& cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libnnvm_compiler.so ${TVM_INSTALL_DIR}/lib
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libtvm.so ${TVM_INSTALL_DIR}/lib
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libtvm_topi.so ${TVM_INSTALL_DIR}/lib
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libvta_fsim.so ${TVM_INSTALL_DIR}/lib
        && cp ${TVM_PREFIX_DIR}/src/extern_tvm/build/libvta_tsim.so ${TVM_INSTALL_DIR}/lib
        && sed -i "s/define\ DMLC_USE_GLOG\ 0/define\ DMLC_USE_GLOG\ 1/g" ${TVM_INSTALL_DIR}/include/dmlc/base.h
    BUILD_IN_SOURCE 1
)

ADD_LIBRARY(tvm SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET tvm PROPERTY IMPORTED_LOCATION ${TVM_LIBRARIES})
ADD_DEFINITIONS(-DPADDLE_WITH_TVM)
ADD_DEPENDENCIES(tvm extern_tvm)