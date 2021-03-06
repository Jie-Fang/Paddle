/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/conv_op.h"

#include <memory>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/platform/cudnn_workspace_helper.h"

namespace paddle {
namespace operators {

void ConvOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                    "Input(Input) of ConvOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Filter"), true,
                    "Input(Filter) of ConvOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Output"), true,
                    "Output(Output) of ConvOp should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");

  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
  std::string padding_algorithm =
      ctx->Attrs().Get<std::string>("padding_algorithm");
  int groups = ctx->Attrs().Get<int>("groups");
  std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");
  const std::string data_format = ctx->Attrs().Get<std::string>("data_format");
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  PADDLE_ENFORCE_EQ(in_dims.size() == 4 || in_dims.size() == 5, true,
                    "Conv intput should be 4-D or 5-D tensor, get %u",
                    in_dims.size());

  PADDLE_ENFORCE_EQ(
      in_dims.size(), filter_dims.size(),
      "Conv input dimension and filter dimension should be the same.");
  PADDLE_ENFORCE_EQ(
      in_dims.size() - strides.size() == 2U, true,
      "Conv input dimension and strides dimension should be consistent.");

  const auto input_channels =
      channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

  PADDLE_ENFORCE_EQ(input_channels, filter_dims[1] * groups,
                    "The number of input channels should be equal to filter "
                    "channels * groups.");
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups, 0,
      "The number of output channels should be divided by groups.");

  framework::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
  }
  framework::DDim filter_data_dims =
      framework::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                           in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (size_t i = 0; i < in_data_dims.size(); ++i) {
    if ((!ctx->IsRuntime()) &&
        (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(ConvOutputSize(in_data_dims[i], filter_dims[i + 2],
                                            dilations[i], paddings[2 * i],
                                            paddings[2 * i + 1], strides[i]));
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }

  ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
  ctx->ShareLoD("Input", "Output");
}

framework::OpKernelType ConvOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  auto input_data_type = ctx.Input<Tensor>("Input")->type();
  std::string data_format =
      "AnyLayout";  // todo enable data layout when it's ready
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
    customized_type_value =
        (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
         input_data_type == framework::DataTypeTrait<uint8_t>::DataType())
            ? kConvMKLDNNINT8
            : kConvMKLDNNFP32;
  }
#endif

  if (input_data_type != framework::proto::VarType::INT8 &&
      input_data_type != framework::proto::VarType::UINT8) {
    auto filter_data_type = ctx.Input<Tensor>("Filter")->type();
    PADDLE_ENFORCE_EQ(input_data_type, filter_data_type,
                      "input and filter data type should be consistent");
  }
  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
                      "float16 can only be used when CUDNN is used");
  }

  auto type = framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                      library, customized_type_value);
#ifdef PADDLE_WITH_CUDA
  std::vector<framework::KernelConfig>& configs = kernel_configs_map_[type];
  // TODO(dangqingqing): Currently conv_fusion_op use cudnn but sets use_cudnn
  // to false. It should be fixed and then here should only create if library
  // is kCUDNN.
  if (configs.empty()) {
    std::shared_ptr<framework::AlgorithmsCache<cudnnConvolutionFwdAlgo_t>> p(
        new framework::AlgorithmsCache<cudnnConvolutionFwdAlgo_t>());
    configs.push_back(p);
  }
#endif
  return type;
}

void Conv2DOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput("Input",
           "(Tensor) The input tensor of convolution operator. "
           "The format of input tensor is NCHW or NHWC, where N is batch size, "
           "C is the "
           "number of channels, H is the height of the feature, "
           "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "H is the height of the filter, and W is the width of the filter. "
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddInput("Bias",
           "(Tensor) Bias to be added to each output of filter application."
           "The format of output tensor is X (one-dimensional) of size equal"
           "to the number of output channels. Only used with MKL-DNN.")
      .AsDispensable();
  AddInput("ResidualData",
           "(Tensor) Tensor with residual data "
           "to which convolution output will be added."
           "Used with fuse_residual_connection fusion.")
      .AsDispensable();
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator. "
            "It has same data fromat and data type as the Input.");
  AddAttr<std::vector<int>>("strides",
                            "(vector<int> default:{1, 1}), the "
                            "strides(h_stride, w_stride) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(vector<int> default:{0, 0}), the "
                            "paddings(pad_height_top, pad_height_bottom, "
                            "pad_width_left, pad_wifth_right)  of "
                            "convolution operator.")
      .SetDefault({0, 0});
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddAttr<int>(
      "groups",
      "(int default:1), the groups number of the convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1}), the "
                            "dilations(h_dilation, w_dilation) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu_before_depthwise_conv",
                "(bool, default false) Only used in cuda depthwise kernel")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("use_quantizer",
                "(bool, default false) "
                "Set to true for operators that should be quantized and use "
                "int8 kernel. "
                "Only used on CPU.")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_brelu",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<float>("fuse_brelu_threshold",
                 "(float, default false 6.0) Only used in mkldnn kernel")
      .SetDefault(6.0f);
  AddAttr<std::string>("fuse_activation",
                       "(string, default \"\") Only used in mkldnn kernel")
      .SetDefault("");
  AddAttr<float>("fuse_alpha",
                 "(float, default 0.0) Only used in mkldnn kernel")
      .SetDefault(0.0f);
  AddAttr<float>("fuse_beta", "(float, default 0.0) Only used in mkldnn kernel")
      .SetDefault(0.0f);
  AddAttr<bool>("fuse_residual_connection",
                "(bool, default false) Only used in mkldnn kernel. Used "
                "whenever convolution output is as an input to residual "
                "connection.")
      .SetDefault(false);
  AddAttr<float>("Scale_in",
                 "Scale_in to be used for int8 input data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(1.0f);
  AddAttr<float>("Scale_out",
                 "Scale_out to be used for int8 output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(1.0f);
  AddAttr<float>("Scale_in_eltwise",
                 "Scale_in_eltwise to be used for int8 eltwise input data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(1.0f);
  AddAttr<std::vector<float>>("Scale_weights",
                              "Scale_weights to be used for int8 weights data."
                              "Only used with MKL-DNN INT8.")
      .SetDefault({1.0f});
  AddAttr<bool>("force_fp32_output",
                "(bool, default false) Force INT8 kernel output FP32, only "
                "used in MKL-DNN INT8")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCHW");
  // TODO(dzhwinter): need to registered layout transform function
  AddAttr<int>("workspace_size_MB",
               "Only used in cudnn kernel. Need set use_cudnn to true."
               "workspace size for cudnn, in MB, "
               "workspace is a section of GPU memory which will be "
               "allocated/freed each time the operator runs, larger "
               "workspace size can increase performance but also requires "
               "better hardware. This size should be chosen carefully.")
      .SetDefault(platform::kDefaultConvWorkspaceSizeLimitMB);
  AddAttr<bool>("exhaustive_search",
                "(bool, default false) cuDNN has many algorithm to calculation "
                "convolution, whether enable exhaustive search "
                "for cuDNN convolution or not, default is False.")
      .SetDefault(false);

  AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and Output(Output) are in NCHW or NHWC format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature.
Filters(Input) is MCHW format. Where M is the number of output image channels, C is
the number of input image channels, H is the height of the filter, and W
is the width of the filter.
Parameters(strides, paddings, dilations) are two elements. These two elements represent
height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
$$
       H_{out}= \frac{(H_{in} + pad_height_top + pad_height_bottom - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + pad_width_left + pad_width_right - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  Apply();
}

void Conv3DOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution operator. "
      "The format of input tensor is NCDHW or NDHWC. Where N is batch size, C "
      "is the "
      "number of channels, D is the depth of the feature, H is the height of "
      "the feature, "
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCDHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "D is the depth of the filter, H is the height of the filter, and W "
           "is the width of the filter."
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddInput("ResidualData",
           "(Tensor) Tensor with residual data "
           "to which convolution output will be added."
           "Used with fuse_residual_connection fusion.")
      .AsDispensable();
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator."
            "It has same data fromat and data type as the Input.");
  AddAttr<std::vector<int>>("strides",
                            "(vector<int>, default:{1, 1, 1}), the "
                            "strides(d_stride, h_stride, w_stride) of "
                            "convolution operator.")
      .SetDefault({1, 1, 1});
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int>, default:{0, 0, 0}), the "
      "paddings(pad_depth_front, pad_depth_back, pad_height_top, "
      "pad_height_bottom, pad_width_left, pad_width_right) of convolution "
      "operator.")
      .SetDefault({0, 0, 0});
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddAttr<int>(
      "groups",
      "(int default:1), the groups number of the convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1, 1}), the "
                            "dilations(d_dilation, h_dilation, w_dilation) of "
                            "convolution operator.")
      .SetDefault({1, 1, 1});
  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<std::string>("fuse_activation",
                       "(string, default \"\") Only used in mkldnn kernel")
      .SetDefault("");
  AddAttr<float>("fuse_alpha",
                 "(float, default 0.0) Only used in mkldnn kernel")
      .SetDefault(0.0f);
  AddAttr<float>("fuse_beta", "(float, default 0.0) Only used in mkldnn kernel")
      .SetDefault(0.0f);
  AddAttr<bool>("fuse_residual_connection",
                "(bool, default false) Only used in mkldnn kernel. Used "
                "whenever convolution output is as an input to residual "
                "connection.")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCDHW) Only used in "
      "An optional string from: \"NDHWC\", \"NCDHW\". "
      "Defaults to \"NDHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCDHW");
  AddAttr<bool>("force_fp32_output",
                "(bool, default false) Only used in mkldnn INT8 kernel")
      .SetDefault(false);
  // TODO(dzhwinter): need to registered layout transform function
  AddAttr<int>("workspace_size_MB",
               "Only used in cudnn kernel. workspace size for cudnn, in MB, "
               "workspace is a section of GPU memory which will be "
               "allocated/freed each time the operator runs, larger "
               "workspace size can increase performance but also requires "
               "better hardware. This size should be chosen carefully.")
      .SetDefault(platform::kDefaultConvWorkspaceSizeLimitMB);
  AddAttr<bool>("exhaustive_search",
                "(bool, default false) cuDNN has many algorithm to calculation "
                "convolution, whether enable exhaustive search "
                "for cuDNN convolution or not, default is False.")
      .SetDefault(false);
  AddComment(R"DOC(
Convolution3D Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and output(Output) are in NCDHW or NDHWC format, where N is batch
size, C is the number of channels,D is the depth of the feature, H is the height of
the feature, and W is the width of the feature.
Filters(Input) is MCDHW format, where M is the number of output image channels,
C is the number of input image channels, D is the depth of the filter,
H is the height of the filter, and W is the width of the filter.
Parameters(strides, paddings, dilations) are three elements. These three elements
represent depth, height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, D_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, D_f, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, D_{out}, H_{out}, W_{out})$
  Where
  $$
       D_{out}= \frac{(D_{in} + pad_depth_front + pad_depth_back - (dilations[0] * (D_f - 1) + 1))}{ strides[0]}+ 1 \\
       H_{out}= \frac{(H_{in} + pad_height_top + pad_height_bottom - (dilations[1] * (H_f - 1) + 1))}{ strides[1]}+ 1 \\
       W_{out}= \frac{(W_{in} + pad_width_left + pad_width_right - (dilations[2] * (W_f - 1) + 1))}{ strides[2]}+ 1
  $$
)DOC");
  Apply();
}

void ConvOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("Filter"))) {
    ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
  }
}

framework::OpKernelType ConvOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library_{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  std::string data_format = "AnyLayout";
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
    customized_type_value = kConvMKLDNNFP32;
  }
#endif

  auto type = framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                      ctx.GetPlace(), layout_, library_,
                                      customized_type_value);
#ifdef PADDLE_WITH_CUDA
  if (library_ == framework::LibraryType::kCUDNN) {
    std::vector<framework::KernelConfig>& configs = kernel_configs_map_[type];
    if (configs.empty()) {
      std::shared_ptr<framework::AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>>
          p(new framework::AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>());
      configs.push_back(p);

      std::shared_ptr<
          framework::AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>>
          p2(new framework::AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>());
      configs.push_back(p2);
    }
  }
#endif
  return type;
}

class Conv2DGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", Input("Input"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput("Bias", Input("Bias"));
    op->SetInput(framework::GradVarName("Output"), OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), InputGrad("Filter"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    op->SetAttrMap(Attrs());

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

class Conv3DGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", Input("Input"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput(framework::GradVarName("Output"), OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), InputGrad("Filter"));

    if (ForwardOp().Inputs().count("ResidualData") != 0) {
      op->SetInput("ResidualData", Input("ResidualData"));
    }

    op->SetAttrMap(Attrs());

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 */
class Conv2DDoubleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType(this->ForwardOpType() + "_grad");
    // I, W, dO, ddI, ddW
    op->SetInput("Input", Input("Input"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput("DOutput", Input(framework::GradVarName("Output")));
    op->SetInput("DDInput", OutputGrad(framework::GradVarName("Input")));
    op->SetInput("DDFilter", OutputGrad(framework::GradVarName("Filter")));

    // ddO, dI, dW
    // Unlike grad op, double grad op does not use name@GRAD@GRAD
    // as key of ops' inputs and outputs.
    auto ddx = OutputGrad(framework::GradVarName("Input"));
    auto ddw = OutputGrad(framework::GradVarName("Filter"));
    std::vector<std::string> empty_str = {};

    op->SetOutput("DDOutput",
                  (ddx.empty() && ddw.empty())
                      ? empty_str
                      : InputGrad(framework::GradVarName("Output")));
    op->SetOutput("DFilter", ddx.empty() ? empty_str : InputGrad("Filter"));
    op->SetOutput("DInput", ddw.empty() ? empty_str : InputGrad("Input"));

    op->SetAttrMap(Attrs());

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 */
class Conv3DDoubleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType(this->ForwardOpType() + "_grad");
    // I, W, dO, ddI, ddW
    op->SetInput("Input", Input("Input"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput("DOutput", Input(framework::GradVarName("Output")));
    op->SetInput("DDInput", OutputGrad(framework::GradVarName("Input")));
    op->SetInput("DDFilter", OutputGrad(framework::GradVarName("Filter")));

    auto ddx = OutputGrad(framework::GradVarName("Input"));
    auto ddw = OutputGrad(framework::GradVarName("Filter"));
    std::vector<std::string> empty_str = {};

    op->SetOutput("DDOutput",
                  (ddx.empty() && ddw.empty())
                      ? empty_str
                      : InputGrad(framework::GradVarName("Output")));
    op->SetOutput("DFilter", ddx.empty() ? empty_str : InputGrad("Filter"));
    op->SetOutput("DInput", ddw.empty() ? empty_str : InputGrad("Input"));

    op->SetAttrMap(Attrs());

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

void ConvOpDoubleGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto x_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("Filter");
  auto do_dims = ctx->GetInputDim("DOutput");

  if (ctx->HasOutput("DDOutput") &&
      (ctx->HasInput("DDInput") || (ctx->HasInput("DDFilter")))) {
    ctx->SetOutputDim("DDOutput", do_dims);
  }
  if (ctx->HasOutput("DFilter") && ctx->HasInput("DDInput")) {
    ctx->SetOutputDim("DFilter", w_dims);
  }
  if (ctx->HasOutput("DInput") && ctx->HasInput("DDFilter")) {
    ctx->SetOutputDim("DInput", x_dims);
  }
}

framework::OpKernelType ConvOpDoubleGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = "AnyLayout";
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
    customized_type_value = kConvMKLDNNFP32;
  }
#endif
  auto type = framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                      ctx.GetPlace(), layout_, library_,
                                      customized_type_value);
#ifdef PADDLE_WITH_CUDA
  if (library_ == framework::LibraryType::kCUDNN) {
    std::vector<framework::KernelConfig>& configs = kernel_configs_map_[type];
    if (configs.empty()) {
      std::shared_ptr<framework::AlgorithmsCache<cudnnConvolutionFwdAlgo_t>> p0(
          new framework::AlgorithmsCache<cudnnConvolutionFwdAlgo_t>());
      configs.push_back(p0);

      std::shared_ptr<
          framework::AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>>
          p1(new framework::AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>());
      configs.push_back(p1);

      std::shared_ptr<framework::AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>>
          p2(new framework::AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>());
      configs.push_back(p2);
    }
  }
#endif
  return type;
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conv2d, ops::ConvOp, ops::Conv2DOpMaker,
                  ops::ConvOpInferVarType, ops::Conv2DGradMaker);
REGISTER_OPERATOR(conv2d_grad, ops::ConvOpGrad, ops::Conv2DDoubleGradMaker);
REGISTER_OPERATOR(conv2d_grad_grad, ops::ConvOpDoubleGrad);

// depthwise convolution op
REGISTER_OPERATOR(depthwise_conv2d, ops::ConvOp, ops::Conv2DOpMaker,
                  ops::ConvOpInferVarType, ops::Conv2DGradMaker);
REGISTER_OPERATOR(depthwise_conv2d_grad, ops::ConvOpGrad);

REGISTER_OPERATOR(conv3d, ops::ConvOp, ops::Conv3DOpMaker,
                  ops::ConvOpInferVarType, ops::Conv3DGradMaker);
REGISTER_OPERATOR(conv3d_grad, ops::ConvOpGrad, ops::Conv3DDoubleGradMaker);
REGISTER_OPERATOR(conv3d_grad_grad, ops::ConvOpDoubleGrad);

// depthwise conv kernel
// TODO(xingzhaolong): neon kernel for mobile
REGISTER_OP_CPU_KERNEL(
    depthwise_conv2d,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    depthwise_conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    conv2d, ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv2d_grad_grad,
    ops::GemmConvDoubleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvDoubleGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    conv3d, ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv3d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv3d_grad_grad,
    ops::GemmConvDoubleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvDoubleGradKernel<paddle::platform::CPUDeviceContext, double>);
