#include <torch/extension.h>

// C++ dispatcher from the .cu
void alopex_step_cuda(at::Tensor param, at::Tensor xstate, double delta, double p_flip, uint64_t seed, uint64_t offset);

void alopex_step(at::Tensor param, at::Tensor xstate, double delta, double p_flip, uint64_t seed, uint64_t offset) {
    alopex_step_cuda(param, xstate, delta, p_flip, seed, offset);
}

at::Tensor init_xstate_like(at::Tensor param) {
    TORCH_CHECK(param.is_cuda(), "param must be CUDA");
    auto x = at::empty_like(param, param.options().dtype(at::kChar), at::MemoryFormat::Contiguous);
    auto r = at::rand_like(param, param.options().dtype(at::kFloat));
    auto s = (r.ge(0.5f).to(at::kChar) * 2 - 1);
    x.copy_(s);
    return x;
}

int64_t philox_offset_increment(int64_t numel) { return (numel + 3) / 4; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alopex_step", &alopex_step, "FlashAlopex fused step (CUDA)",
          pybind11::arg("param"), pybind11::arg("xstate"),
          pybind11::arg("delta"), pybind11::arg("p_flip"),
          pybind11::arg("seed"), pybind11::arg("offset"));
    m.def("init_xstate_like", &init_xstate_like, "Init xstate (+1/-1) like param (CUDA)");
    m.def("philox_offset_increment", &philox_offset_increment, "Philox offset increment for numel");
}