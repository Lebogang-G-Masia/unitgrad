#include "include/matrix.hpp"
#include <memory>
#include <variant>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <unordered_set>

#ifndef UNIT_GRAD_HPP
#define UNIT_GRAD_HPP

namespace UnitGrad {
    template <typename U>
    class UnitTensor {
        public:
            using Ptr = std::shared_ptr<UnitTensor<U>>;

            U data;
            U grad;
            std::vector<Ptr> prev {};
            std::string op;
            std::function<void()> _backward = [](){};

            UnitTensor(U data_, std::vector<Ptr> children={}, std::string op_="") : data(data_), prev(children), op(op_) {
                auto shape = data.shape();
                grad = U(shape[0], shape[1]);
            };

            static Ptr make(U data_) {
                return std::make_shared<UnitTensor<U>>(data_);
            }

            friend Ptr operator+(const Ptr& l, const Ptr& r) {
                Ptr out = UnitTensor<U>::make(l->data + r->data);
                out->prev = {l, r};
                out->op = "+";

                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [l, r, out_ptr] {
                    l->grad += out_ptr->grad;
                    r->grad += out_ptr->grad;
                };
                return out;
            }

            friend Ptr operator-(const Ptr& l, const Ptr& r) {
                Ptr out = UnitTensor<U>::make(l->data - r->rdata);
                out->prev = {l, r};
                out->op = "-";
                
                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [l, r, out_ptr] {
                    l->grad += out_ptr->grad;
                    r->grad -= out_ptr->grad;
                };
            }

            friend Ptr operator*(const Ptr& l, const Ptr& r) {
                Ptr out = UnitTensor<U>::make(l->data * r->data);
                out->prev = {l, r};
                out->op = "matmul";

                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [l, r, out_ptr] {
                    l->grad += out_ptr->grad.matmul(r->data.transpose());
                    r->grad += l->data.transpose().matmul(out_ptr->grad);
                };
                return out;
            }

            friend Ptr relu(const Ptr& u) {
                auto relu_func = [](float x) -> float { return x > 0 ? x : 0.0f; };
                Ptr out = UnitTensor<U>::make(u->data.map(relu_func));
                out->prev = {u};
                out->op = "ReLU";

                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [u, out_ptr] {
                    auto d_relu = [](float x) -> float { return x > 0 ? 1.0f : 0.0f; };
                    U deriv_mat = u->data.map(d_relu);
                    u->grad += out_ptr->grad.element_wise(deriv_mat);
                };
                return out;
            }

            friend void backward(const Ptr& root) {
                std::vector<Ptr> topo {};
                std::unordered_set<Ptr> visited {};

                std::function<void(const Ptr&)> build_topo = [&](const Ptr& p) {
                    if (p == nullptr) return;
                    if (visited.find(p) == visited.end()) {
                        visited.insert(p);
                        for (const Ptr& child: p->prev) build_topo(child);
                        topo.push_back(p);
                    }
                };
                build_topo(root);

                auto shape = root->data.shape();
                std::size_t total_elements = shape[0] * shape[1];
                for (std::size_t i = 0; i < total_elements; i++) {
                    root->grad.data_ptr()[i] = 1.0f;
                }
                std::reverse(topo.begin(), topo.end());
                for (const Ptr& child: topo) child->_backward();
            }
    };
}

#endif // UNIT_GRAD_HPP
