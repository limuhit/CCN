#include "main.hpp"
#include <torch/extension.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    py::class_<dtow_opt>(m,"DtowOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &dtow_opt::to)
        .def("forward", &dtow_opt::forward_cuda)
        .def("backward", &dtow_opt::backward_cuda);

    py::class_<quant_opt>(m,"QuantOp")
        .def(py::init<int, int, float, int, int, float, int, bool>())
        .def("to", &quant_opt::to)
        .def("forward", &quant_opt::quant_forward_cuda)
        .def("backward", &quant_opt::quant_backward_cuda);

    py::class_<context_reshape_opt>(m,"ContextReshapeOp")
        .def(py::init<int, int, bool>())
        .def("to", &context_reshape_opt::to)
        .def("forward", &context_reshape_opt::forward_cuda)
        .def("backward", &context_reshape_opt::backward_cuda);

    py::class_<entropy_gmm_opt>(m,"EntropyGmmOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &entropy_gmm_opt::to)
        .def("forward", &entropy_gmm_opt::forward_cuda)
        .def("backward", &entropy_gmm_opt::backward_cuda);

    py::class_<mask_constrain_opt>(m,"MaskConstrainOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &mask_constrain_opt::to)
        .def("forward", &mask_constrain_opt::forward_cuda)
        .def("backward", &mask_constrain_opt::backward_cuda);

    py::class_<d_input_opt>(m,"DInputOp")
        .def(py::init<int, int, bool>())
        .def("to", &d_input_opt::to)
        .def("forward", &d_input_opt::forward_cuda)
        .def("backward", &d_input_opt::backward_cuda);


    py::class_<d_extract_opt>(m,"DExtractOp")
        .def(py::init<bool, int, bool>())
        .def("to", &d_extract_opt::to)
        .def("forward", &d_extract_opt::forward_cuda)
        .def("backward", &d_extract_opt::backward_cuda);

    py::class_<d_output_opt>(m,"DOutputOp")
        .def(py::init<int, float, int, bool>())
        .def("to", &d_output_opt::to)
        .def("forward", &d_output_opt::forward_cuda)
        .def("backward", &d_output_opt::backward_cuda);

    py::class_<dquant_opt>(m,"DquantOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &dquant_opt::to)
        .def("forward", &dquant_opt::forward_cuda)
        .def("backward", &dquant_opt::backward_cuda);

    py::class_<dconv_opt>(m,"DconvOp")
        .def(py::init<int, int, int, int, int, int, bool>())
        .def("to", &dconv_opt::to)
        .def("forward", &dconv_opt::forward_cuda)
        .def("backward", &dconv_opt::backward_cuda);
};