#include <concepts>
#include <type_traits>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <ostream>
#include <algorithm>
#include <unordered_set>

#ifndef UNIT_GRAD_H
#define UNIT_GRAD_H

namespace UnitGrad {
    template <typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    template <Numeric U>
    class UnitTensor {
        public:
            using Ptr = std::shared_ptr<UnitTensor<U>>;
    
            U data;
            U grad = 0;
            std::vector<Ptr> prev {};
            std::string op;
            std::function<void()> _backward = [](){};
    
            UnitTensor(U data_, std::vector<Ptr> children={}, std::string op_="") : data(data_), prev(children), op(op_) {};
    
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
                Ptr out = UnitTensor<U>::make(l->data - r->data);
                out->prev = {l, r};
                out->op = "-";
    
                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [l, r, out_ptr] {
                    l->grad -= out_ptr->grad;
                    r->grad -= out_ptr->grad;
                };
                return out;
            }
 
    
            friend Ptr operator*(const Ptr& l, const Ptr& r) {
                Ptr out = UnitTensor<U>::make(l->data * r->data);
                out->prev = {l, r};
                out->op = "*";
    
                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [l, r, out_ptr] {
                    l->grad += r->data * out_ptr->grad;
                    r->grad += l->data * out_ptr->grad;
                };
                return out;
            }
    
            friend Ptr relu(const Ptr& u) {
                Ptr out = UnitTensor<U>::make(std::max(static_cast<U>(0), u->data));
                out->prev = {u};
                out->op = "ReLU";
    
                UnitTensor<U>* out_ptr = out.get();
                out->_backward = [u, out_ptr] {
                    u->grad += out_ptr->data > 0 ? out_ptr->grad : 0;
                };
                return out;
            }
    
            friend std::ostream& operator<<(std::ostream& out, const Ptr& u) {
                out << "Unit(data=" << u->data << ", grad=" << u->grad << ')';
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
                root->grad = static_cast<U>(1);
                std::reverse(topo.begin(), topo.end());
                for (const Ptr& child: topo) child->_backward();
            }
    };
}

#endif // UNIT_GRAD_H
