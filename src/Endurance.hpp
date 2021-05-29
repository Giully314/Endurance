#pragma once

#include <concepts>
#include <memory>
#include <array>
#include <iostream>
#include <utility>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <numeric>
#include "NestedInitList.hpp"

namespace endurance
{

    //BASIC TENSOR STRUCTS

    /*
    TensorRaw represents the raw contigous data. It's "meaningless" without a TensorView that gives interpretation specifying dimensions. 
    Row major.
    */
    template <typename T, size_t Size>
    requires (std::integral<T> || std::floating_point<T>) /*Maybe i can add that T is a group as concept*/
    struct TensorRaw 
    {
        using type = T;
        static constexpr size_t size = Size;

        constexpr TensorRaw() : data(std::make_shared<std::array<T, Size>>()) { }

        constexpr TensorRaw(std::initializer_list<T> elems) : data(std::make_shared<std::array<T, Size>>())
        {
            std::copy(elems.begin(), elems.end(), data->begin());
        }

        std::shared_ptr<std::array<type, size>> data; //maybe shared_ptr is better?
    };
    

    /*
    TensorView gives interpretation to a TensorRaw object by defining the "dimensions" of the tensor.
    */
    template <size_t ...Dimensions> 
    struct TensorView
    {
        //Static constexpr?
        static constexpr std::array<size_t, sizeof...(Dimensions)> dims = {Dimensions...};
    };

    /*
    TODO:
    Check the problems with std::initializer_list (unnecessary copies). In case solve the problem.
    */
    template <typename T, size_t ...Dimensions>
    struct Tensor
    {
        constexpr Tensor() = default;

        //TODO: check Dimensions and # of elements in the NestedInitList
        constexpr Tensor(meta::NestedInitList<T, sizeof...(Dimensions)> elems)
        { 
            ind = 0;
            init<sizeof...(Dimensions)>(std::move(elems));
        }



        TensorRaw<T, (Dimensions * ...)> tensor; 
        TensorView<Dimensions...> shape;

        template <size_t Levels> 
        constexpr void init(meta::NestedInitList<T, Levels> l)
        {
            if constexpr(Levels == 1ULL)
            {
                auto it = l.begin();   
                while (it != l.end())
                {
                    (*tensor.data)[ind++] = *it;
                    ++it;
                }
                return;
            }

            if constexpr(Levels > 1)
            {    
                auto it = l.begin();   
                while (it != l.end())
                {
                    init<Levels - 1>(*it);
                    ++it;
                }
            }
        }
    private:
        int ind = 0;
    };

    // template <typename T, size_t ...Dimensions> 
    // Tensor(meta::NestedInitList<T, Dimensions...>) -> Tensor<T, Dimensions...>;

    //END BASIC TENSOR STRUCTS


    //GENERAL METHOD
    


    //END GENERAL METHOD


    //OPERATOR OVERLOAD FOR TENSORS
    

    template <size_t ...Dimensions>
    std::ostream& operator<<(std::ostream& os, const TensorView<Dimensions...>& v)
    {
        auto it = v.dims.cbegin();
        while (it != v.dims.cend())
        {
            os << *it++ << " ";
        }
        return os;
    }


    //Da rendere ricorsiva
    template <typename T, size_t ...Dimensions>
    std::ostream& operator<<(std::ostream& os, const Tensor<T, Dimensions...>& t)
    {
        for (int i = 0; i < t.tensor.size; ++i)
        {
            os << (*t.tensor.data)[i] << " ";
        }
        return os;
    }
    //END OPERATOR OVERLOAD FOR TENSORS

} //endurance