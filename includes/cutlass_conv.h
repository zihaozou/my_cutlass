#pragma once
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/threadblock/threadblock_swizzle.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/device_nhwc_padding.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include <memory>
#define CUTLASS_CHECK(status)                                                                          \
    {                                                                                                  \
        cutlass::Status error = status;                                                                \
        if (error != cutlass::Status::kSuccess)                                                        \
        {                                                                                              \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                                                    \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }
#define MY_CUTLASS_HALF_PRECISION (!(MY_CUTLASS_MIN_GPU_ARCH == 61 || MY_CUTLASS_MIN_GPU_ARCH <= 52))
using Precision_t = std::conditional_t<MY_CUTLASS_HALF_PRECISION, cutlass::half_t, float>;

using SmArch = std::conditional_t<MY_CUTLASS_MIN_GPU_ARCH >= 80,
                                  std::conditional_t<std::is_same<Precision_t, float>::value, cutlass::arch::Sm75,
                                                     cutlass::arch::Sm80>,
                                  std::conditional_t<MY_CUTLASS_MIN_GPU_ARCH >= 75,
                                                     cutlass::arch::Sm75, cutlass::arch::Sm70>>;
using TypeAccumulator = std::conditional_t<std::is_same<SmArch, cutlass::arch::Sm80>::value, cutlass::half_t, float>;
using TypeCompute = std::conditional_t<std::is_same<SmArch, cutlass::arch::Sm80>::value, cutlass::half_t, float>;

template <typename T>
using MMAOp = typename std::conditional < std::is_same<T, float>::value ||
              MY_CUTLASS_MIN_GPU_ARCH<70, cutlass::arch::OpClassSimt,
                                      cutlass::arch::OpClassTensorOp>::type;
template <typename T>
using ShapeMMAOp = typename std::conditional<
    std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value,
    typename std::conditional<
        std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::gemm::GemmShape<8, 8, 4>>::type,
    cutlass::gemm::GemmShape<1, 1, 1>>::type;

template <typename ThreadBlock, typename warp>
struct TBWarpShapeConfig
{
    using kThreadBlock = ThreadBlock;
    using kWarp = warp;
};

using TBWarpSplitKShape = typename std::conditional<
    std::is_same<MMAOp<Precision_t>, cutlass::arch::OpClassSimt>::value,
    TBWarpShapeConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
    TBWarpShapeConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>>::type;

using TBWarpShape = typename std::conditional<
    std::is_same<MMAOp<Precision_t>, cutlass::arch::OpClassSimt>::value,
    TBWarpShapeConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
    TBWarpShapeConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>>::type;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, int InputAlignment, int OutputAlignment>
using ConvNtoMforwardKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    TypeA,
    LayoutA,
    TypeB, LayoutB,
    TypeC, LayoutC,
    TypeAccumulator,
    MMAOp<TypeA>,
    SmArch,
    typename TBWarpShapeConfig::kThreadBlock,
    typename TBWarpShapeConfig::kWarp,
    ShapeMMAOp<TypeA>,
    cutlass::epilogue::thread::LinearCombination<
        TypeC,
        OutputAlignment,
        TypeAccumulator,
        TypeCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    (InputAlignment % 8) ? cutlass::conv::IteratorAlgorithm::kFixedChannels : cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    InputAlignment, InputAlignment>::Kernel;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv2to8ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv2to4ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value / 2>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv2to2ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value / 4>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv4to8ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv4to4ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value / 2>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv4to2ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value / 4>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv8to8ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv8to4ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value / 2>>;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv8to2ForwardOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value / 4>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, int InputAlignment, int OutputAlignment, cutlass::conv::StrideSupport Stride>
using ConvNtoMBackwardDataKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    TypeA,
    LayoutA,
    TypeB, LayoutB,
    TypeC, LayoutC,
    TypeAccumulator,
    MMAOp<TypeA>,
    SmArch,
    typename TBWarpShapeConfig::kThreadBlock,
    typename TBWarpShapeConfig::kWarp,
    ShapeMMAOp<TypeA>,
    cutlass::epilogue::thread::LinearCombination<
        TypeC,
        OutputAlignment,
        TypeAccumulator,
        TypeCompute>,
    std::conditional_t<Stride == cutlass::conv::StrideSupport::kUnity, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    Stride,
    InputAlignment,
    OutputAlignment>::Kernel;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv2to2BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value / 4, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv2to4BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value / 2, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv2to8BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2, 128 / cutlass::sizeof_bits<TypeC>::value, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv4to2BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value / 4, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv4to4BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value / 2, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv4to8BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4, 128 / cutlass::sizeof_bits<TypeC>::value, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv8to2BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value / 4, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv8to4BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value / 2, Stride>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, cutlass::conv::StrideSupport Stride>
using Conv8to8BackwardDataOp = cutlass::conv::device::ImplicitGemmConvolution<ConvNtoMBackwardDataKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 128 / cutlass::sizeof_bits<TypeA>::value, 128 / cutlass::sizeof_bits<TypeC>::value, Stride>>;

template <typename Type, typename Layout>
cutlass::HostTensor<Type, Layout> PadChannel(cutlass::HostTensor<Type, Layout> &src, int pad_channels, cudaStream_t stream = nullptr)
{
    cutlass::Tensor4DCoord padded_dim(src.extent().n(), src.extent().h(), src.extent().w(), pad_channels);
    cutlass::HostTensor<Type, Layout> padded(padded_dim);
    cutlass::nhwc_padding(src.extent(), padded_dim, src.device_ref(), padded.device_ref(), stream);
    return padded;
}

template <typename Conv, typename TypeC, typename LayoutC, typename TypeD, typename LayoutD>
void ConvImpl(typename Conv::Arguments &args, cutlass::HostTensor<TypeC, LayoutC> &tensor_c, cutlass::HostTensor<TypeD, LayoutD> &tensor_d, cudaStream_t stream = nullptr)
{
    using EpilogueOp = typename Conv::EpilogueOutputOp;
    using ReductionOp = cutlass::reduction::thread::ReduceAdd<
        TypeAccumulator,
        typename EpilogueOp::ElementAccumulator,
        EpilogueOp::kCount>;
    using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
        cutlass::MatrixShape<4, 32 * EpilogueOp::kCount>,
        EpilogueOp,
        ReductionOp>;
    using ReductionDevice = cutlass::reduction::device::ReduceSplitK<ReductionKernel>;
    using ReductionStrideIndex = typename ReductionDevice::StrideIndex;

    Conv convOp;
    size_t workspace_size = convOp.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = convOp.can_implement(args);
    CUTLASS_CHECK(status);
    status = convOp.initialize(args, workspace.get());
    CUTLASS_CHECK(status);
    if (args.split_k_mode == cutlass::conv::SplitKMode::kParallel)
    {
        args.ref_D.reset(reinterpret_cast<typename Conv::ElementC *>(workspace.get()));
        status = convOp.update(args, workspace.get());
        CUTLASS_CHECK(status);
    }

    status = convOp(stream);
    CUTLASS_CHECK(status);
    if (args.split_k_mode == cutlass::conv::SplitKMode::kParallel)
    {
        ReductionDevice reduction_op;
        typename ReductionDevice::Arguments reduction_args(
            cutlass::conv::implicit_gemm_problem_size(Conv::kConvolutionalOperator, args.problem_size).mn(),
            args.problem_size.split_k_slices,
            cutlass::conv::implicit_gemm_tensor_c_size(Conv::kConvolutionalOperator, args.problem_size),
            {reinterpret_cast<TypeAccumulator *>(workspace.get()),
             ReductionStrideIndex(tensor_c.stride()[Conv::ImplicitGemmKernel::kTensorCStrideIdx])},
            {tensor_d.device_data(),
             ReductionStrideIndex(tensor_d.stride()[Conv::ImplicitGemmKernel::kTensorCStrideIdx])},
            {tensor_c.device_data(),
             ReductionStrideIndex(tensor_c.stride()[Conv::ImplicitGemmKernel::kTensorCStrideIdx])},
            // apply alpha, beta to obtain the following equation alpha * ReduceAdd(A * B) + beta * C
            {TypeCompute(1.0), TypeCompute(0.0)});
        status = reduction_op.initialize(reduction_args);
        CUTLASS_CHECK(status);
        status = reduction_op(stream);
    }
}

#define CHECK_CHANNEL_(c) ((c) % 8 == 0) ? 0 : (((c) == 4) ? 1 : (((c) == 2) ? 2 : 3))
#define RUN_CONV_DIFF_CHAN_FORWARD(x)                                                               \
    typename Conv##x##F::Arguments args{                                                            \
        problem_size,                                                                               \
        tensor_a.device_ref(),                                                                      \
        tensor_b.device_ref(),                                                                      \
        tensor_c.device_ref(),                                                                      \
        tensor_d.device_ref(),                                                                      \
        {TypeCompute(1), TypeCompute(0)},                                                           \
        (split_k > 1) ? cutlass::conv::SplitKMode::kParallel : cutlass::conv::SplitKMode::kSerial}; \
    ConvImpl<Conv##x##F>(args, tensor_c, tensor_d, stream)
int i = 1;
constexpr uint choose_op(uint x, uint y)
{
    return (1 << (CHECK_CHANNEL_(x) + 4)) | (1 << (CHECK_CHANNEL_(y)));
}
template <typename Config, cutlass::conv::Mode mode, typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
void ConvForward(cutlass::HostTensor<TypeA, LayoutA> &tensor_a, cutlass::HostTensor<TypeB, LayoutB> &tensor_b, cutlass::HostTensor<TypeC, LayoutC> &tensor_c, cutlass::HostTensor<TypeD, LayoutD> &tensor_d, cutlass::Tensor4DCoord &padding, cutlass::MatrixCoord &conv_stride, cutlass::MatrixCoord &conv_dilation, int split_k = 1, cudaStream_t stream = nullptr)
{
    using Conv22F = Conv2to2ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv24F = Conv2to4ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv28F = Conv2to8ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv42F = Conv4to2ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv44F = Conv4to4ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv48F = Conv4to8ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv82F = Conv8to2ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv84F = Conv8to4ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
    using Conv88F = Conv8to8ForwardOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;

    static_assert(std::is_same<TypeA, TypeB>::value, "Type of A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of C and D must be equal");
    static_assert(std::is_same<LayoutC, LayoutD>::value, "Layout C and D must be equal");

    cutlass::conv::Conv2dProblemSize problem_size(
        tensor_a.extent(),
        tensor_b.extent(),
        padding,
        conv_stride,
        conv_dilation,
        tensor_d.extent(),
        mode,
        split_k);
    std::unordered_map<uint, std::function<void()>> conv_map{
        {choose_op(8, 8), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(88); }},
        {choose_op(8, 4), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(84); }},
        {choose_op(8, 2), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(82); }},
        {choose_op(4, 8), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(48); }},
        {choose_op(4, 4), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(44); }},
        {choose_op(4, 2), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(42); }},
        {choose_op(2, 8), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(28); }},
        {choose_op(2, 4), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(24); }},
        {choose_op(2, 2), [&]
         { RUN_CONV_DIFF_CHAN_FORWARD(22); }}};
    conv_map[choose_op(tensor_b.extent().c(), tensor_b.extent().n())]();
}

#define RUN_CONV_DIFF_CHAN_BACKWARD(x, stride)                                                      \
    typename Conv##x##BD##stride ::Arguments args{                                                  \
        problem_size,                                                                               \
        tensor_a.device_ref(),                                                                      \
        tensor_b.device_ref(),                                                                      \
        tensor_c.device_ref(),                                                                      \
        tensor_d.device_ref(),                                                                      \
        {TypeCompute(1), TypeCompute(0)},                                                           \
        (split_k > 1) ? cutlass::conv::SplitKMode::kParallel : cutlass::conv::SplitKMode::kSerial}; \
    ConvImpl<Conv##x##BD##stride>(args, tensor_c, tensor_d, stream)

template <typename Config, cutlass::conv::Mode mode, typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
void ConvBackwardData(cutlass::HostTensor<TypeA, LayoutA> &tensor_a, cutlass::HostTensor<TypeB, LayoutB> &tensor_b, cutlass::HostTensor<TypeC, LayoutC> &tensor_c, cutlass::HostTensor<TypeD, LayoutD> &tensor_d, cutlass::Tensor4DCoord &padding, cutlass::MatrixCoord &conv_stride, cutlass::MatrixCoord &conv_dilation, int split_k = 1, cudaStream_t stream = nullptr)
{
    using Conv22BDS = Conv2to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv24BDS = Conv2to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv28BDS = Conv2to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv42BDS = Conv4to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv44BDS = Conv4to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv48BDS = Conv4to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv82BDS = Conv8to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv84BDS = Conv8to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv88BDS = Conv8to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kStrided>;
    using Conv22BDU = Conv2to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv24BDU = Conv2to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv28BDU = Conv2to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv42BDU = Conv4to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv44BDU = Conv4to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv48BDU = Conv4to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv82BDU = Conv8to2BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv84BDU = Conv8to4BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    using Conv88BDU = Conv8to8BackwardDataOp<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config, cutlass::conv::StrideSupport::kUnity>;
    static_assert(std::is_same<TypeA, TypeB>::value, "Type of A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of C and D must be equal");
    static_assert(std::is_same<LayoutC, LayoutD>::value, "Layout C and D must be equal");
    cutlass::conv::Conv2dProblemSize problem_size(
        tensor_d.extent(),
        tensor_b.extent(),
        padding,
        conv_stride,
        conv_dilation,
        tensor_a.extent(),
        mode,
        split_k);
    if (conv_stride.row() == 1 && conv_stride.column() == 1)
    {
        std::unordered_map<uint, std::function<void()>>
            conv_map{
                {choose_op(8, 8), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(88, U); }},
                {choose_op(8, 4), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(84, U); }},
                {choose_op(8, 2), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(82, U); }},
                {choose_op(4, 8), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(48, U); }},
                {choose_op(4, 4), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(44, U); }},
                {choose_op(4, 2), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(42, U); }},
                {choose_op(2, 8), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(28, U); }},
                {choose_op(2, 4), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(24, U); }},
                {choose_op(2, 2), [&]
                 { RUN_CONV_DIFF_CHAN_BACKWARD(22, U); }}};
        conv_map[choose_op(tensor_b.extent().n(), tensor_b.extent().c())]();
    }
    else
    {
        std::unordered_map<uint, std::function<void()>> conv_map{
            {choose_op(8, 8), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(88, S); }},
            {choose_op(8, 4), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(84, S); }},
            {choose_op(8, 2), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(82, S); }},
            {choose_op(4, 8), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(48, S); }},
            {choose_op(4, 4), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(44, S); }},
            {choose_op(4, 2), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(42, S); }},
            {choose_op(2, 8), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(28, S); }},
            {choose_op(2, 4), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(24, S); }},
            {choose_op(2, 2), [&]
             { RUN_CONV_DIFF_CHAN_BACKWARD(22, S); }}};
        conv_map[choose_op(tensor_b.extent().n(), tensor_b.extent().c())]();
    }
}