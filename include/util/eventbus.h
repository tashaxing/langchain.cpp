// langchain/util/eventbus.h
// Signal/slot + EventBus event dispatch.
//
// Two complementary primitives for in-process event dispatch:
//
//   1. Signal<Args...> — Qt-style typed signal with multiple slots.
//      Each connect() returns a SlotId; call disconnect() or let the
//      ScopedConnection RAII handle do it. Emission is synchronous on
//      the calling thread; slot exceptions are caught and logged.
//
//   2. EventBus — process-wide, string-keyed pub/sub for loosely coupled
//      modules that don't share a type. Payload is std::any. Implementation
//      lives in eventbus.cpp.
//
// Both are thread-safe.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <any>

namespace langchain
{
namespace util
{

using SlotId = std::uint64_t;

// ---------------- typed Signal ----------------

template <typename... Args>
class Signal
{
public:
    using Slot = std::function<void(Args...)>;

    Signal()  = default;
    ~Signal() = default;

    Signal(const Signal&)            = delete;
    Signal& operator=(const Signal&) = delete;

    SlotId connect(Slot s)
    {
        std::lock_guard<std::mutex> lk(mu_);
        SlotId id = ++next_id_;
        slots_.emplace_back(id, std::move(s));
        return id;
    }

    bool disconnect(SlotId id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = slots_.begin(); it != slots_.end(); ++it)
        {
            if (it->first == id)
            {
                slots_.erase(it);
                return true;
            }
        }
        return false;
    }

    void disconnect_all()
    {
        std::lock_guard<std::mutex> lk(mu_);
        slots_.clear();
    }

    std::size_t slot_count() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return slots_.size();
    }

    // Synchronously call every connected slot on the emitting thread.
    // Slots are snapshotted under the lock then invoked unlocked, so a
    // slot may safely (dis)connect during emission without deadlock.
    void emit(Args... args) const
    {
        std::vector<std::pair<SlotId, Slot>> snapshot;
        {
            std::lock_guard<std::mutex> lk(mu_);
            snapshot = slots_;
        }
        for (const auto& kv : snapshot)
        {
            try
            {
                kv.second(args...);
            }
            catch (const std::exception& e)
            {
                std::cerr << "[eventbus] slot threw: " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "[eventbus] slot threw unknown exception\n";
            }
        }
    }

    // operator() is a convenience alias for emit().
    void operator()(Args... args) const
    {
        emit(args...);
    }

private:
    mutable std::mutex                              mu_;
    std::vector<std::pair<SlotId, Slot>>            slots_;
    SlotId                                          next_id_{0};
};

// RAII helper that disconnects on destruction. Move-only.
template <typename SignalT>
class ScopedConnection
{
public:
    ScopedConnection() = default;
    ScopedConnection(SignalT* sig, SlotId id)
        : sig_(sig), id_(id)
    {
    }
    ~ScopedConnection()
    {
        reset();
    }

    ScopedConnection(const ScopedConnection&)            = delete;
    ScopedConnection& operator=(const ScopedConnection&) = delete;

    ScopedConnection(ScopedConnection&& other) noexcept
        : sig_(other.sig_), id_(other.id_)
    {
        other.sig_ = nullptr;
        other.id_  = 0;
    }
    ScopedConnection& operator=(ScopedConnection&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            sig_       = other.sig_;
            id_        = other.id_;
            other.sig_ = nullptr;
            other.id_  = 0;
        }
        return *this;
    }

    void reset()
    {
        if (sig_ && id_ != 0)
        {
            sig_->disconnect(id_);
        }
        sig_ = nullptr;
        id_  = 0;
    }

    explicit operator bool() const
    {
        return sig_ != nullptr && id_ != 0;
    }

private:
    SignalT* sig_{nullptr};
    SlotId   id_{0};
};

// ---------------- EventBus (process-wide, string topic) ----------------

// Lightweight publish/subscribe with std::any payloads. For loosely coupled
// modules; for hot paths, prefer a typed Signal. Implementation lives in
// eventbus.cpp so we don't pull <any> machinery into every TU that imports a
// typed Signal.
class EventBus
{
public:
    using Handler = std::function<void(const std::any&)>;

    static EventBus& instance();

    SlotId subscribe(const std::string& topic, Handler h);
    bool   unsubscribe(SlotId id);
    void   publish(const std::string& topic, const std::any& payload);
    std::size_t subscriber_count(const std::string& topic) const;
    void   clear();

private:
    EventBus() = default;

    struct Entry
    {
        SlotId      id;
        std::string topic;
        Handler     handler;
    };

    mutable std::mutex mu_;
    std::vector<Entry> entries_;
    SlotId             next_id_{0};
};

} // namespace util
} // namespace langchain
