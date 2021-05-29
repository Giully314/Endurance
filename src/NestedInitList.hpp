#pragma once


#include <initializer_list>


namespace endurance
{
    namespace meta
    {
        template <typename T, size_t Levels>
        struct nested_init_list
        {
            using type = std::initializer_list<typename nested_init_list<T, Levels - 1>::type>;
        };

        template <typename T>
        struct nested_init_list<T, 0>
        {
            using type = T;
        };


        template <typename T, size_t Levels>
        using NestedInitList = typename nested_init_list<T, Levels>::type;
    }
}