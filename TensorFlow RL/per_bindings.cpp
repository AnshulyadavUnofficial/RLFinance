#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sumTree.hpp"

namespace py = pybind11;

using PER = PrioritizedExperienceReplayBuffer<py::object>;

// Explicit instantiation for py::object
template class PrioritizedExperienceReplayBuffer<py::object>;
template std::vector<float> stochastic_priority_replacement<py::object>(
    SumTree<py::object>&,
    const std::vector<std::pair<py::object, float>>&
);

PYBIND11_MODULE(per_buffer, m) {
    m.doc() = "Prioritized Experience Replay Buffer (C++ backend)";

    py::class_<PER>(m, "PrioritizedExperienceReplayBuffer")
        .def(py::init<int, float, float>(), 
            py::arg("capacity"), 
            py::arg("alpha") = 0.6f, 
            py::arg("beta") = 0.7f)
        .def("add_batch_experience", &PER::add_batch_experience)
        .def("sample", &PER::sample)
        .def("update_leaf_priorities", &PER::update_leaf_priorities)
        .def("set_beta", &PER::set_beta)
        .def("get_beta", &PER::get_beta)
        .def_readonly("capacity", &PER::capacity)   // expose read-only
        .def_readonly("length", &PER::length);      // expose read-only
}
