// activate.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// Enumerations for fast C++ callbacks
enum class ActFn  { SIGMOID, RELU, TANH };
enum class AggFn  { SUM, MAX, MEAN };

// C++ implementations of activations & aggregations
static inline double cpp_sigmoid(double x) { return 1.0/(1.0+std::exp(-x)); }
static inline double cpp_relu(double x)    { return x>0?x:0; }
static inline double cpp_tanh(double x)    { return std::tanh(x); }
static inline double cpp_sum(const std::vector<double>& v) {
    double s=0; for(auto x:v) s+=x; return s;
}
static inline double cpp_max(const std::vector<double>& v) {
    return v.empty()?0:*std::max_element(v.begin(),v.end());
}
static inline double cpp_mean(const std::vector<double>& v) {
    return v.empty()?0:cpp_sum(v)/v.size();
}

// Plain struct for a node
struct NodeEvalC {
    int node;
    ActFn act;
    AggFn agg;
    double bias, resp;
    std::vector<std::pair<int,double>> links;
};

class CPPNet {
public:
    std::vector<int>       input_nodes, output_nodes;
    std::vector<NodeEvalC> evs;

    // one‐time build
    CPPNet(py::list inputs_py,
           py::list outputs_py,
           std::vector<py::tuple> node_evals_py)
    {
        for(auto o: inputs_py)  input_nodes .push_back(o.cast<int>());
        for(auto o: outputs_py) output_nodes.push_back(o.cast<int>());

        // import the exact Python functions
        py::module_ neat_act = py::module_::import("neat.activations");
        py::module_ neat_agg = py::module_::import("neat.aggregations");
        auto py_sig  = neat_act.attr("sigmoid_activation");
        auto py_relu = neat_act.attr("relu_activation");
        auto py_tan  = neat_act.attr("tanh_activation");
        auto py_sum  = neat_agg.attr("sum_aggregation");
        auto py_max  = neat_agg.attr("max_aggregation");
        auto py_mean = neat_agg.attr("mean_aggregation");

        // translate each tuple
        for(auto &ev: node_evals_py) {
            NodeEvalC c;
            c.node = ev[0].cast<int>();

            // activation
            py::object actf = ev[1].cast<py::object>();
            if      (actf.equal(py_sig))  c.act = ActFn::SIGMOID;
            else if (actf.equal(py_relu)) c.act = ActFn::RELU;
            else if (actf.equal(py_tan))  c.act = ActFn::TANH;
            else throw std::runtime_error("Unknown activation");

            // aggregation
            py::object aggf = ev[2].cast<py::object>();
            if      (aggf.equal(py_sum))  c.agg = AggFn::SUM;
            else if (aggf.equal(py_max))  c.agg = AggFn::MAX;
            else if (aggf.equal(py_mean)) c.agg = AggFn::MEAN;
            else throw std::runtime_error("Unknown aggregation");

            c.bias  = ev[3].cast<double>();
            c.resp  = ev[4].cast<double>();
            c.links = ev[5].cast<std::vector<std::pair<int,double>>>();
            evs.push_back(std::move(c));
        }
    }

    // pure‐C++ activate using an unordered_map
    std::vector<double> activate(const std::vector<double>& inputs) {
        if(inputs.size() != input_nodes.size())
            throw std::runtime_error("Input size mismatch");

        std::unordered_map<int,double> vals;
        vals.reserve(input_nodes.size() + evs.size());

        // write inputs
        for(size_t i=0;i<inputs.size();++i)
            vals[input_nodes[i]] = inputs[i];

        // node evaluations
        for(auto &c: evs) {
            std::vector<double> acc;
            acc.reserve(c.links.size());
            for(auto &lk: c.links)
                acc.push_back(vals[lk.first] * lk.second);

            double agg_out;
            switch(c.agg) {
              case AggFn::SUM:  agg_out = cpp_sum(acc);  break;
              case AggFn::MAX:  agg_out = cpp_max(acc);  break;
              case AggFn::MEAN: agg_out = cpp_mean(acc); break;
            }

            double pre  = c.bias + c.resp * agg_out;
            double post;
            switch(c.act) {
              case ActFn::SIGMOID: post = cpp_sigmoid(pre); break;
              case ActFn::RELU:    post = cpp_relu(pre);    break;
              case ActFn::TANH:    post = cpp_tanh(pre);    break;
            }

            vals[c.node] = post;
        }

        // collect outputs
        std::vector<double> out;
        out.reserve(output_nodes.size());
        for(int idx: output_nodes)
            out.push_back(vals[idx]);  // missing keys => 0.0
        return out;
    }
};

PYBIND11_MODULE(_cppneat, m) {
    py::class_<CPPNet>(m, "CPPNet")
        .def(py::init<py::list, py::list, std::vector<py::tuple>>())
        .def("activate", &CPPNet::activate);
}
