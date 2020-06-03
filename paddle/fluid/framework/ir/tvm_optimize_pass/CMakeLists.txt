if (WITH_GPU)
    cc_library(tvm_optimize_pass SRCS tvm_optimize_pass.cc DEPS fusion_group tvm)
    cc_test(test_tvm_optimize_pass SRCS tvm_optimize_pass_tester.cc DEPS tvm_optimize_pass graph_viz_pass tvm)
endif()
