#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/device_memory.h"

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
using TypeAccumulator = std::conditional_t<std::is_same<Precision_t, float>::value, float, cutlass::half_t>;
using TypeCompute = std::conditional_t<std::is_same<Precision_t, float>::value, float, cutlass::half_t>;

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

template <typename Config>
using SwizzleThreadBlock = std::conditional_t<
    std::is_same<Config, TBWarpSplitKShape>::value, cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig, int alignment>
using ConvNChanforwardKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    TypeA, LayoutA,
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
        128 / cutlass::sizeof_bits<TypeC>::value,
        TypeAccumulator,
        TypeCompute>,
    SwizzleThreadBlock<TBWarpShapeConfig>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    alignment, alignment>::Kernel;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv4ChanForward = cutlass::conv::device::ImplicitGemmConvolution<ConvNChanforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 4>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using Conv2ChanForward = cutlass::conv::device::ImplicitGemmConvolution<ConvNChanforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig, 2>>;

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using ConvChanforwardKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    TypeA, LayoutA,
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
        128 / cutlass::sizeof_bits<TypeC>::value,
        TypeAccumulator,
        TypeCompute>,
    SwizzleThreadBlock<TBWarpShapeConfig>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutA, typename LayoutB, typename LayoutC, typename TBWarpShapeConfig>
using ConvChanForward = cutlass::conv::device::ImplicitGemmConvolution<ConvChanforwardKernel<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, TBWarpShapeConfig>>;

template <class Conv>
void conv_forward_impl(const typename Conv::Arguments &args)
{
    Conv convOp;
    size_t workspace_size = convOp.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = convOp.can_implement(args);
    CUTLASS_CHECK(status);
    status = convOp.initialize(args, workspace.get());
    CUTLASS_CHECK(status);
    status = convOp();
    CUTLASS_CHECK(status);
}

template <typename Config, cutlass::conv::Mode mode, typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
void conv_forward(cutlass::HostTensor<TypeA, LayoutA> &tensorA, cutlass::HostTensor<TypeB, LayoutB> &tensorB, cutlass::HostTensor<TypeC, LayoutC> &tensorC, cutlass::HostTensor<TypeD, LayoutD> &tensorD, cutlass::Tensor4DCoord &padding, cutlass::MatrixCoord &convStride, cutlass::MatrixCoord &convDilation)
{
    static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
    static_assert(std::is_same<LayoutC, LayoutD>::value, "Layout of matrix C and D must be equal");

    cutlass::conv::Conv2dProblemSize problem_size(tensorA.extent(), tensorB.extent(), padding, convStride, convDilation, tensorD.extent(), mode, 1);
    if (tensorA.extent().c() <= 2)
    {
        using conv2 = Conv2ChanForward<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
        typename conv2::Arguments args(problem_size, tensorA.device_ref(), tensorB.device_ref(), tensorC.device_ref(), tensorD.device_ref(), {TypeCompute(1.0), TypeCompute(0.0)});
        conv_forward_impl<conv2>(args);
    }
    else if (tensorA.extent().c() <= 4 && tensorA.extent().c() > 2)
    {
        using conv4 = Conv4ChanForward<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
        typename conv4::Arguments args(problem_size, tensorA.device_ref(), tensorB.device_ref(), tensorC.device_ref(), tensorD.device_ref(), {TypeCompute(1.0), TypeCompute(0.0)});
        conv_forward_impl<conv4>(args);
    }
    else
    {
        using conv = ConvChanForward<TypeA, TypeB, TypeC, LayoutA, LayoutB, LayoutC, Config>;
        typename conv::Arguments args(problem_size, tensorA.device_ref(), tensorB.device_ref(), tensorC.device_ref(), tensorD.device_ref(), {TypeCompute(1.0), TypeCompute(0.0)});
        conv_forward_impl<conv>(args);
    }
}