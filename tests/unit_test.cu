#include <gtest/gtest.h>
#include "cutlass_conv.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/device_nhwc_padding.h"
using ActivationType = cutlass::half_t;
using FilterType = cutlass::half_t;
using OutputType = cutlass::half_t;
static int size = 8;
TEST(ConvTest, ConvolutionFoward_2to32)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 2);
    cutlass::Tensor4DCoord filter_size(2, 3, 3, 2);
    cutlass::Tensor4DCoord output_size(1, size, size, 2);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionFoward_4to32)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 4);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 4);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionFoward_32to2)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(2, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 2);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionFoward_32to4)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(4, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 4);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionFoward_32to32)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionFoward_32to32_k_Split)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output_ref(output_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_output.host_view());
    cutlass::reference::host::TensorFill(tensor_output_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvForward<TBWarpSplitKShape, cutlass::conv::Mode::kConvolution>(tensor_activation, tensor_filter, tensor_output, tensor_output, padding, stride, dilation, 2);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dFprop(problem_size, tensor_activation.host_ref(), tensor_filter.host_ref(), tensor_output_ref.host_ref(), tensor_output_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_output.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_output.host_view(), tensor_output_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Unity_32to2)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 2);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 2);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Unity_32to4)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 4);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 4);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Unity_32to32)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_32to32_Unity_k_Split)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpSplitKShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 2);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Stride_32to2)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 2);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 2);
    cutlass::Tensor4DCoord output_size(1, (size - 1) / 3 + 1, (size - 1) / 3 + 1, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(3, 3);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Stride_32to4)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 4);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 4);
    cutlass::Tensor4DCoord output_size(1, (size - 1) / 3 + 1, (size - 1) / 3 + 1, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(3, 3);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Data_Stride_32to32)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(32, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, (size - 1) / 3 + 1, (size - 1) / 3 + 1, 32);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(3, 3);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_activation_ref(activation_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_filter.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_activation.host_view());
    cutlass::reference::host::TensorFill(tensor_activation_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardData<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_filter, tensor_activation, tensor_activation, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dDgrad(problem_size, tensor_output.host_ref(), tensor_filter.host_ref(), tensor_activation_ref.host_ref(), tensor_activation_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_activation.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_activation.host_view(), tensor_activation_ref.host_view());
    ASSERT_TRUE(passed);
}

TEST(ConvTest, ConvolutionBackward_Weight_Unity_32to2)
{
    cutlass::Tensor4DCoord activation_size(1, size, size, 32);
    cutlass::Tensor4DCoord filter_size(2, 3, 3, 32);
    cutlass::Tensor4DCoord output_size(1, size, size, 2);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);

    cutlass::HostTensor<ActivationType, cutlass::layout::TensorNHWC> tensor_activation(activation_size);
    cutlass::HostTensor<FilterType, cutlass::layout::TensorNHWC> tensor_filter(filter_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_output(output_size);
    cutlass::HostTensor<OutputType, cutlass::layout::TensorNHWC> tensor_filter_ref(filter_size);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_activation.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(tensor_output.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_filter.host_view());
    cutlass::reference::host::TensorFill(tensor_filter_ref.host_view());
    tensor_activation.sync_device();
    tensor_filter.sync_device();
    tensor_output.sync_device();
    ConvBackwardWeight<TBWarpShape, cutlass::conv::Mode::kConvolution>(tensor_output, tensor_activation, tensor_filter, tensor_filter, padding, stride, dilation, 1);
    cutlass::conv::Conv2dProblemSize problem_size(activation_size, filter_size, padding, stride, dilation, output_size, cutlass::conv::Mode::kConvolution, 1);
    cutlass::reference::host::Conv2dWgrad(problem_size, tensor_output.host_ref(), tensor_activation.host_ref(), tensor_filter_ref.host_ref(), tensor_filter_ref.host_ref(), TypeCompute(1.0), TypeCompute(0.0));
    tensor_filter.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(tensor_filter.host_view(), tensor_filter_ref.host_view());
    ASSERT_TRUE(passed);
}