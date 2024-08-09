#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "solvers/sqp/sqp_pcg.cuh"

namespace py = pybind11;

//TODO: pcg_config and d_dynMem_const
py::tuple py_sqpSolvePcg(py::array_t<float> eePos_goal_traj,
    py::array_t<float> xu,
    py::array_t<float> lambda,
    float rho,
    float rho_reset) {

}