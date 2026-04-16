#include "runtime/dsl/compiled_ops.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_narrow(const CompiledOp& op) {
    if (op.inputs.size() != 1) {
        throw std::runtime_error("dispatch_narrow: expected exactly one input");
    }
    if (op.outputs.size() != 1) {
        throw std::runtime_error("dispatch_narrow: expected exactly one output");
    }

    Tensor& in = resolve_tensor(op.inputs[0]);
    if (in.Rank <= 0) {
        throw std::runtime_error("dispatch_narrow: input rank must be > 0");
    }

    const int rank = in.Rank;
    int dim = op.attrs.split_concat_dim;  // reuses the common "dim" attr
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) {
        throw std::runtime_error("dispatch_narrow: dim out of range");
    }

    const int start = op.attrs.narrow_start;
    const int length = op.attrs.narrow_length;
    const long in_dim = in.Sizes[dim];

    if (start < 0 || length <= 0 || start + length > in_dim) {
        std::ostringstream oss;
        oss << "dispatch_narrow: invalid range [" << start << ", " << start + length
            << ") for dim " << dim << " of size " << in_dim;
        throw std::runtime_error(oss.str());
    }

    // Compute output shape (same rank, narrowed along dim)
    std::vector<long> out_shape(in.Sizes.begin(), in.Sizes.begin() + rank);
    out_shape[dim] = length;

    // Compute byte offset and strides
    const std::size_t elem_bytes = get_dtype_size(in.DType);
    long inner = 1;
    for (int i = dim + 1; i < rank; ++i) {
        inner *= in.Sizes[i];
    }
    long outer = 1;
    for (int i = 0; i < dim; ++i) {
        outer *= in.Sizes[i];
    }

    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.DType != in.DType || !tensor_shape_matches(out, out_shape)) {
        out = mRunState.temp_alloc(in.DType, out_shape, "narrow_output");
        mTemps.push_back(out);
    }

    const std::byte* in_ptr = static_cast<const std::byte*>(in.Data);
    std::byte* out_ptr = static_cast<std::byte*>(out.Data);

    const std::size_t src_pitch =
        static_cast<std::size_t>(in_dim) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::size_t row_bytes =
        static_cast<std::size_t>(length) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::byte* src_base =
        in_ptr + static_cast<std::size_t>(start) * static_cast<std::size_t>(inner) * elem_bytes;

    if (outer == 1) {
        CUDA_CHECK(cudaMemcpyAsync(
            out_ptr, src_base, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            out_ptr, row_bytes,
            src_base, src_pitch,
            row_bytes, static_cast<std::size_t>(outer),
            cudaMemcpyDeviceToDevice,
            mRunState.MainStream));
    }

    store_tensor(op.outputs[0], out);
}

}  // namespace dsl
