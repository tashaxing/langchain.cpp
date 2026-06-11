// langchain/util/logging.cpp

#include "util/logging.h"

#include <algorithm>
#include <vector>

namespace langchain
{
namespace log
{

namespace
{

// Global logger instance + mutex.
std::mutex g_mutex;
std::shared_ptr<spdlog::logger> g_logger;

// ---------------------------------------------------------------------------
// Reversed rotating file sink
//
// spdlog's built-in rotating_file_sink keeps the *oldest* slice at index 0
// and pushes newer ones to higher indices.  We want the opposite: index 0 is
// always the *current* (newest) file, and older slices have larger indices.
//
// Naming: base.log (current), base.1.log, base.2.log, ...
// On rotation: base.log -> base.1.log, base.1.log -> base.2.log, ...
// ---------------------------------------------------------------------------
template <typename Mutex>
class reversed_rotating_file_sink final : public spdlog::sinks::base_sink<Mutex>
{
public:
    reversed_rotating_file_sink(std::string base_filename,
                                std::size_t max_size,
                                std::size_t max_files)
        : base_filename_(std::move(base_filename)),
          max_size_(max_size),
          max_files_(max_files),
          current_size_(0)
    {
        if (max_size_ == 0)
            throw spdlog::spdlog_ex("reversed_rotating_file_sink: max_size cannot be zero");
        if (max_files_ > 200000)
            throw spdlog::spdlog_ex("reversed_rotating_file_sink: max_files too large");

        file_helper_.open(base_filename_);
        current_size_ = file_helper_.size();
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);
        auto new_size = current_size_ + formatted.size();

        if (new_size > max_size_) {
            file_helper_.flush();
            if (file_helper_.size() > 0) {
                rotate_();
                new_size = formatted.size();
            }
        }
        file_helper_.write(formatted);
        current_size_ = new_size;
    }

    void flush_() override
    {
        file_helper_.flush();
    }

private:
    std::string base_filename_;
    std::size_t max_size_;
    std::size_t max_files_;
    std::size_t current_size_;
    spdlog::details::file_helper file_helper_;

    static std::string calc_filename(const std::string& base, std::size_t index)
    {
        if (index == 0)
            return base;
        std::string basename, ext;
        std::tie(basename, ext) = spdlog::details::file_helper::split_by_extension(base);
        return basename + "." + std::to_string(index) + ext;
    }

    void rotate_()
    {
        file_helper_.close();

        // Shift existing slices upward: N -> N+1 (delete the oldest if over limit).
        for (std::size_t i = max_files_; i > 0; --i) {
            std::string src  = calc_filename(base_filename_, i - 1);
            std::string dest = calc_filename(base_filename_, i);

            if (!spdlog::details::os::path_exists(src))
                continue;

            // Remove destination if it exists, then rename.
            (void)spdlog::details::os::remove(dest);
            if (spdlog::details::os::rename(src, dest) != 0) {
                // On Windows high-rotation can race with AV; retry once.
                spdlog::details::os::sleep_for_millis(100);
                (void)spdlog::details::os::remove(dest);
                if (spdlog::details::os::rename(src, dest) != 0) {
                    file_helper_.reopen(true);
                    current_size_ = 0;
                    throw spdlog::spdlog_ex("reversed_rotating_file_sink: rotate failed");
                }
            }
        }

        file_helper_.reopen(true);
        current_size_ = 0;
    }
};

using reversed_rotating_file_sink_mt = reversed_rotating_file_sink<std::mutex>;

} // anonymous namespace

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
void init(const std::string& log_dir,
          const std::string& log_filename,
          spdlog::level::level_enum level,
          std::size_t max_file_size,
          std::size_t max_files,
          bool console)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    std::filesystem::path dir(log_dir);
    if (!std::filesystem::exists(dir))
        std::filesystem::create_directories(dir);

    std::string full_path = (dir / log_filename).string();

    auto file_sink = std::make_shared<reversed_rotating_file_sink_mt>(
        full_path, max_file_size, max_files);

    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(file_sink);

    if (console) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks.push_back(console_sink);
    }

    auto dist_sink = std::make_shared<spdlog::sinks::dist_sink_mt>();
    dist_sink->set_sinks(sinks);

    // Pattern:
    //   [2026-06-01 12:34:56.123456789] [12345] [INFO ] [file.cpp:42] message
    // Flags:
    //   %Y-%m-%d %H:%M:%S.%F   date + time + nanoseconds
    //   %t                     thread id
    //   %l                     level name
    //   %s:%#                  short source filename : line number
    //   %v                     message
    dist_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%t] [%l] [%s:%#] %v");

    auto logger = std::make_shared<spdlog::logger>("langchain", dist_sink);
    logger->set_level(level);
    logger->flush_on(spdlog::level::info);

    g_logger = logger;
    spdlog::register_logger(g_logger);
}

// ---------------------------------------------------------------------------
// set_level
// ---------------------------------------------------------------------------
void set_level(spdlog::level::level_enum level)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logger)
        g_logger->set_level(level);
}

// ---------------------------------------------------------------------------
// get
// ---------------------------------------------------------------------------
spdlog::logger& get()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logger)
        return *g_logger;

    // Fallback: stderr-only logger so we never crash on uninit usage.
    static auto fallback = []()
    {
        auto s = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        s->set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%t] [%l] [%s:%#] %v");
        auto l = std::make_shared<spdlog::logger>("langchain_fallback", s);
        l->set_level(spdlog::level::info);
        return l;
    }();
    return *fallback;
}

} // namespace log
} // namespace langchain
