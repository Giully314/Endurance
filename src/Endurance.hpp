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

        type& operator[](size_t index)
        {
            return (*data)[index];
        }

        const type& operator[](size_t index) const
        {
            return (*data)[index];
        }

        /*
        Shared_ptr to avoid copies and to share the underlying data.
        */
        std::shared_ptr<std::array<type, size>> data; 
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



    template <typename T, size_t ...Dimensions>
    class Tensor
    {
    public:
        constexpr Tensor() = default;

        //TODO: check Dimensions and # of elements in the NestedInitList
        constexpr Tensor(meta::NestedInitList<T, sizeof...(Dimensions)> elems)
        { 
            init<sizeof...(Dimensions)>(std::move(elems));
        }


    private:
        TensorView<Dimensions...> shape;
        TensorRaw<T, (Dimensions * ...)> tensor; 


        template <size_t Levels>
        constexpr void init(const meta::NestedInitList<T, Levels>& l)
        {
            int i = 0;
            from_nestedlist_to_tensor<Levels>(l, i);
        }

        /*
        TODO: check if NestedInitList does unnecessary copies.
        */
        template <size_t Levels> 
        constexpr void from_nestedlist_to_tensor(const meta::NestedInitList<T, Levels>& l, int& curr_index)
        {
            if constexpr(Levels == 1ULL)
            {
                auto it = l.begin();   
                while (it != l.end())
                {
                    tensor[curr_index++] = *it;
                    ++it;
                }
                return;
            }

            if constexpr(Levels > 1)
            {    
                auto it = l.begin();   
                while (it != l.end())
                {
                    from_nestedlist_to_tensor<Levels - 1>(*it, curr_index);
                    ++it;
                }
            }
        }

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