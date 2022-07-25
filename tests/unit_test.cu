#include <gtest/gtest.h>
#include "cutlass_conv.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/device_nhwc_padding.h"
using InputType = cutlass::half_t;
using FilterType = cutlass::half_t;
using OutputType = cutlass::half_t;

TEST(ConvTest, ConvolutionFoward_2to32)
{
    cutlass::Tensor4DCoord input_size(1, 256, 256, 2);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 2);
    cutlass::Tensor4DCoord output_size(1, 256, 256, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<InputType, cutlass::layout::TensorNHWC> tensor_input(input_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_input.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_input.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_input, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(input_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_input.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_FALSE(passed);
}

TEST(ConvTest, ConvolutionFoward_4to32)
{
    cutlass::Tensor4DCoord input_size(1, 256, 256, 4);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 4);
    cutlass::Tensor4DCoord output_size(1, 256, 256, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<InputType, cutlass::layout::TensorNHWC> tensor_input(input_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_input.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_input.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_input, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(input_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_input.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_FALSE(passed);
}
TEST(ConvTest, ConvolutionFoward_1to32)
{
    cutlass::Tensor4DCoord input_size(1, 256, 256, 1);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 1);
    cutlass::Tensor4DCoord output_size(1, 256, 256, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<InputType, cutlass::layout::TensorNHWC> tensor_input(input_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_input.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_input.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_input, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(input_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_input.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_FALSE(passed);
}