// langchain/util/singleton.h
// Header-only Meyers singleton wrapper. Thread-safe initialization (C++11+),
// no manual lock needed. Derive your class and friend Singleton<T> so its
// private ctor is reachable, or use it directly with a default-constructible T.
//
// Usage:
//   class Config { friend class langchain::util::Singleton<Config>;
//                  Config() = default; ... };
//   auto& c = langchain::util::Singleton<Config>::instance();
#pragma once

#include <utility>

namespace langchain
{
namespace util
{

template <typename T>
class Singleton
{
public:
    // The canonical, default-constructed instance.
    static T& instance()
    {
        static T inst;
        return inst;
    }

    // Variant for types needing constructor arguments on first call.
    // Subsequent calls ignore the arguments and return the same instance.
    template <typename... Args>
    static T& instance_with(Args&&... args)
    {
        static T inst(std::forward<Args>(args)...);
        return inst;
    }

    Singleton()                            = delete;
    ~Singleton()                           = delete;
    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;
};

} // namespace util
} // namespace langchain
