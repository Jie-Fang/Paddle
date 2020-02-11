if (WITH_GPU)
    cc_library(tvm_optimize_pass SRCS tvm_optimize_pass.cc DEPS fusion_group tvm)
endif()
