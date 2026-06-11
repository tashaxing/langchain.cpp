// src/util/eventbus.cpp — EventBus implementation.
#include "util/eventbus.h"
#include "util/logging.h"

namespace langchain
{
namespace util
{

EventBus& EventBus::instance()
{
    static EventBus inst;
    return inst;
}

SlotId EventBus::subscribe(const std::string& topic, Handler h)
{
    std::lock_guard<std::mutex> lk(mu_);
    SlotId id = ++next_id_;
    entries_.push_back(Entry{id, topic, std::move(h)});
    return id;
}

bool EventBus::unsubscribe(SlotId id)
{
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
    {
        if (it->id == id)
        {
            entries_.erase(it);
            return true;
        }
    }
    return false;
}

void EventBus::publish(const std::string& topic, const std::any& payload)
{
    std::vector<Handler> snapshot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : entries_)
        {
            if (e.topic == topic)
            {
                snapshot.push_back(e.handler);
            }
        }
    }
    for (auto& h : snapshot)
    {
        try
        {
            h(payload);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("[eventbus] handler for '{}' threw: {}", topic, e.what());
        }
        catch (...)
        {
            LOG_ERROR("[eventbus] handler for '{}' threw unknown exception", topic);
        }
    }
}

std::size_t EventBus::subscriber_count(const std::string& topic) const
{
    std::lock_guard<std::mutex> lk(mu_);
    std::size_t n = 0;
    for (const auto& e : entries_)
    {
        if (e.topic == topic)
        {
            ++n;
        }
    }
    return n;
}

void EventBus::clear()
{
    std::lock_guard<std::mutex> lk(mu_);
    entries_.clear();
}

} // namespace util
} // namespace langchain
