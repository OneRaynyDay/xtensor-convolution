// utilities
#include <string>
#include <typeinfo>
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#include <iostream>
#include <vector>

// xtensor
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xslice.hpp>

// xtensor-blas
#include <xtensor-blas/xlinalg.hpp>

#define OPTIMIZED

using namespace xt;
using placeholders::_;
int padding = 1;

// globals
const int IN_LEN = 4;
const int H_IDX = 2;
const int W_IDX = 3;
const int C_IDX = 1;
const int K_IDX = 0;
const int N_IDX = 0;

/// INTERMEDIATE VALUE DEF:
/// Input = (N, C, H, W)
/// Filter = (K, C, R, S)
/// Output = (N, K, P, Q)
template <typename T, typename O>
auto conv2d(const xt::xexpression<T>& data,
            const xt::xexpression<O>& filter,
            int strides = 1,
            int padding = 1
           ){
    using value_type = std::common_type_t<typename T::value_type, typename O::value_type>;

    // Convention: _x for pre-resizing via padding.
    xt::xarray<value_type> _x = xt::eval(data.derived_cast());
    xt::xarray<value_type> f = xt::eval(filter.derived_cast());

    auto&& _x_shape = _x.shape();
    auto&& f_shape = f.shape();

    if(_x_shape.size() != IN_LEN || f_shape.size() != IN_LEN)
        throw std::runtime_error("conv2d: Shapes mismatch.");

    // Pad x with padding on all 4 sides with `padding` # of zeros.
    // Generate zeros with the shape of x_shape + 2*padding on height and width dimensions
    auto x_shape = _x_shape; // copy
    x_shape[H_IDX] += 2*padding;
    x_shape[W_IDX] += 2*padding;

    // Not necessary to perform convolution, but will lead to hard-to-debug
    // errors due to leftovers missing
    XTENSOR_ASSERT(x_shape[H_IDX] % strides == 0)
    XTENSOR_ASSERT(x_shape[W_IDX] % strides == 0)
    XTENSOR_ASSERT(x_shape[C_IDX] == f_shape[C_IDX]);

    // Can I make this lazy? Attempts to make this auto fails
    xt::xarray<value_type, T::static_layout> x = xt::zeros<value_type>(x_shape);

    /// Calculated from strides
    xt::view(x, 
        xt::all(), 
        xt::all(), 
        xt::range(padding, _x_shape[H_IDX] + padding), 
        xt::range(padding, _x_shape[W_IDX] + padding)) = _x;
    
    auto N = x_shape[N_IDX];
    auto H = x_shape[H_IDX];
    auto W = x_shape[W_IDX];
    auto K = f_shape[K_IDX];
    auto C = f_shape[C_IDX];
    auto R = f_shape[H_IDX];
    auto S = f_shape[W_IDX];
    auto P = (x_shape[H_IDX] - f_shape[H_IDX])/strides + 1;
    auto Q = (x_shape[W_IDX] - f_shape[W_IDX])/strides + 1;

#ifdef OPTIMIZED
    std::conditional_t<(T::static_layout == O::static_layout) &&
                       (T::static_layout != layout_type::dynamic),
                       xarray<value_type, T::static_layout>,
                       xarray<value_type, XTENSOR_DEFAULT_LAYOUT>> im2col;

    im2col.resize({N, P, Q, C*R*S});

    // Perform flatten filter - cheap because already alligned.
    f.reshape({K, C*R*S});

    // TODO: Finish this
    // Perform im2col
    for(auto i = 0; i <= H - R; i += strides){
        for(auto j = 0; j <= W - S; j += strides){
            // A chunk of (N, C, R, S). Transpose to (C, R, S, N) then reshape into (C*R*S, N)
            auto v = xt::eval(xt::view(x, xt::all(), xt::all(), xt::range(i, i+R), xt::range(j, j+S)));
            // Perform flatten filter - cheap because already alligned.
            v.reshape({N, C*R*S});
            // Assign this to im2col:  
            auto x = i / strides;
            auto y = j / strides;
            xt::view(im2col, xt::all(), x, y, xt::all()) = v;
        }
    }
    // Perform flatten im2col - cheap because already alligned.
    im2col.reshape({N*P*Q, C*R*S});
    // pre-shaped result
    auto _result = xt::linalg::dot(im2col, xt::transpose(f)); // Get out N*P*Q, K
    _result.reshape({N, P, Q, K}); // cheap reshape
    // Need to call eval() or else it breaks.
    auto result = xt::eval(xt::transpose(_result, {N_IDX, 3, 1, 2})); // expensive transpose
#else
    std::conditional_t<(T::static_layout == O::static_layout) &&
                       (T::static_layout != layout_type::dynamic),
                       xarray<value_type, T::static_layout>,
                       xarray<value_type, XTENSOR_DEFAULT_LAYOUT>> result;

    result.resize({N, K, P, Q});
    for(auto i = 0; i <= H - R; i += strides){
        for(auto j = 0; j <= W - S; j += strides){
            // yields a N x C x R x S slice
            auto v = xt::view(x, xt::all(), xt::all(), xt::range(i, i+R), xt::range(j, j+S));
            for(auto k = 0; k < K; k++){
                // Yields a 1 x C x R x S slice of filter
                auto f_slice = xt::view(f, k, xt::all(), xt::all(), xt::all());
                // Reduction along height and width and channel to give us
                // A single N x C x R x S slice reduced into N
                // which is assigned to result(:, k, i', j')

                // TODO: Try xt::sum(m, {axes}, xt::evaluation_strategy::immediate{});
                auto prod = xt::sum(v * f_slice, {C_IDX, H_IDX, W_IDX});
                auto x = i / strides;
                auto y = j / strides;
                xt::view(result, xt::all(), k, x, y) = prod;
            }
        }
    }
#endif
    return result;
}

int main(){
    xt::xarray<double> a1
    {
    {
        {
            {2, 3},
            {3, 4},
        },
    },
    {
        {
            {2, 1},
            {3, 2},
        },
    }
    };

    // (3,1,2,2)
    xt::xarray<double> a2
    {
    {
        {
            {2, 1},
            {1, 2},
        }
    },
    {
        {
            {3, 4},
            {3, 2},
        }
    },
    {
        {
            {3, 5},
            {5, 2},
        }
    }
    };

    auto&& x = conv2d(a1, a2, 2, 0);
    std::cout << x << std::endl;
}
