// llamacpp in-process backend.
// Only compiled when LC_ENABLE_LLAMA=ON (gated by CMakeLists.txt).
//
// Implementation notes:
//   * Uses llama_chat_apply_template to format messages with the model's
//     bundled template (falls back to "chatml" if absent).
//   * Sampler chain: top_k -> top_p -> temp -> dist (or greedy if temp == 0).
//   * Streaming feeds each detokenized piece to the StreamCallback.
//   * Token positions are tracked automatically by llama_decode via
//     llama_batch_get_one (the simplest correct path for non-batched use).
#include "llm/llamacpp_llm.h"

#if defined(LC_HAS_LLAMA)
#include <llama.h>
#endif

#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <vector>

namespace langchain
{
namespace llm
{

struct LlamacppLLM::Impl
{
#if defined(LC_HAS_LLAMA)
    struct BackendRuntime
    {
        BackendRuntime()
        {
            llama_backend_init();
        }

        ~BackendRuntime()
        {
            llama_backend_free();
        }
    };

    struct ModelDeleter
    {
        void operator()(llama_model* model) const
        {
            if (model)
            {
                llama_model_free(model);
            }
        }
    };

    struct ContextDeleter
    {
        void operator()(llama_context* ctx) const
        {
            if (ctx)
            {
                llama_free(ctx);
            }
        }
    };

    BackendRuntime backend;
    std::unique_ptr<llama_model, ModelDeleter> model;
    std::unique_ptr<llama_context, ContextDeleter> ctx;
    const llama_vocab* vocab = nullptr;
    std::string        tmpl_override; // user-provided chat template; empty => model's
#endif
};

#if defined(LC_HAS_LLAMA)

namespace
{

// Format messages with llama.cpp's bundled template. Resizes buffer as needed.
std::string apply_chat_template(const llama_model* model,
                                const std::string& tmpl_override,
                                const std::vector<Message>& msgs,
                                bool add_assistant_open)
{
    std::vector<llama_chat_message> chat;
    chat.reserve(msgs.size());
    for (const auto& m : msgs)
    {
        chat.push_back({to_string(m.role), m.content.c_str()});
    }

    const char* tmpl = tmpl_override.empty() ? nullptr : tmpl_override.c_str();
    if (!tmpl)
    {
        // Try the model's default template; if missing, fall back to chatml.
        const char* model_tmpl = llama_model_chat_template(model, /*name=*/nullptr);
        tmpl = model_tmpl ? model_tmpl : "chatml";
    }

    std::string buf(2048, '\0');
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                          add_assistant_open, buf.data(),
                                          static_cast<int32_t>(buf.size()));
    if (n < 0)
    {
        throw LCError("LlamacppLLM: llama_chat_apply_template failed");
    }
    if (static_cast<std::size_t>(n) > buf.size())
    {
        buf.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                      add_assistant_open, buf.data(),
                                      static_cast<int32_t>(buf.size()));
        if (n < 0)
        {
            throw LCError("LlamacppLLM: llama_chat_apply_template failed (resize)");
        }
    }
    buf.resize(static_cast<std::size_t>(n));
    return buf;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text,
                                  bool add_special)
{
    int32_t n_neg = -llama_tokenize(vocab, text.c_str(),
                                    static_cast<int32_t>(text.size()),
                                    nullptr, 0, add_special,
                                    /*parse_special=*/true);
    std::vector<llama_token> out(static_cast<std::size_t>(n_neg));
    int32_t n = llama_tokenize(vocab, text.c_str(),
                               static_cast<int32_t>(text.size()),
                               out.data(), static_cast<int32_t>(out.size()),
                               add_special, /*parse_special=*/true);
    if (n < 0)
    {
        throw LCError("LlamacppLLM: tokenize failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return out;
}

std::string detokenize_piece(const llama_vocab* vocab, llama_token tok)
{
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0,
                                     /*special=*/false);
    if (n < 0)
    {
        return std::string();
    }
    return std::string(buf, buf + n);
}

struct SamplerDeleter
{
    void operator()(llama_sampler* sampler) const
    {
        if (sampler)
        {
            llama_sampler_free(sampler);
        }
    }
};

using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

SamplerPtr build_sampler(const ChatRequest& req, std::uint32_t seed)
{
    auto sparams = llama_sampler_chain_default_params();
    SamplerPtr smpl(llama_sampler_chain_init(sparams));
    float temp = req.temperature.value_or(0.7f);
    if (temp <= 0.0f)
    {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());
    }
    else
    {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl.get(),
            llama_sampler_init_top_p(req.top_p.value_or(0.95f), 1));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(seed));
    }
    return smpl;
}

// Drive the decode/sample loop. on_piece is called per detokenized chunk;
// return false from it to abort.
ChatResponse generate(llama_context* ctx,
                      const llama_vocab* vocab,
                      llama_model* model,
                      const std::string& tmpl_override,
                      const ChatRequest& req,
                      std::uint32_t seed,
                      const std::function<bool(const std::string&)>& on_piece)
{
    auto prompt = apply_chat_template(model, tmpl_override, req.messages,
                                      /*add_assistant_open=*/true);
    auto tokens = tokenize(vocab, prompt, /*add_special=*/true);
    if (tokens.empty())
    {
        throw LCError("LlamacppLLM: empty tokenization");
    }

    auto smpl = build_sampler(req, seed);

    // Prefill.
    llama_batch batch = llama_batch_get_one(tokens.data(),
                                            static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx, batch) != 0)
    {
        throw LCError("LlamacppLLM: prefill llama_decode failed");
    }

    Usage usage;
    usage.prompt_tokens = static_cast<int>(tokens.size());

    std::string output;
    int max_new = req.max_tokens.value_or(512);
    std::string finish_reason = "stop";
    bool aborted = false;

    for (int i = 0; i < max_new; ++i)
    {
        llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
        llama_sampler_accept(smpl.get(), id);

        if (llama_vocab_is_eog(vocab, id))
        {
            break;
        }

        std::string piece = detokenize_piece(vocab, id);
        output += piece;
        usage.completion_tokens++;

        if (!on_piece(piece))
        {
            aborted = true;
            finish_reason = "abort";
            break;
        }

        // Caller-supplied stop strings.
        bool stop_hit = false;
        for (const auto& s : req.stop)
        {
            if (!s.empty() && output.size() >= s.size() &&
                output.compare(output.size() - s.size(), s.size(), s) == 0)
            {
                output.resize(output.size() - s.size());
                stop_hit = true;
                break;
            }
        }
        if (stop_hit)
        {
            finish_reason = "stop";
            break;
        }

        // Feed back this token for the next position.
        llama_batch next = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, next) != 0)
        {
            finish_reason = "error";
            break;
        }

        if (i == max_new - 1)
        {
            finish_reason = "length";
        }
    }

    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;

    ChatResponse out;
    out.message = Message::assistant(std::move(output));
    out.finish_reason = finish_reason;
    out.usage = usage;
    (void)aborted;
    return out;
}

} // namespace

#endif // LC_HAS_LLAMA

LlamacppLLM::LlamacppLLM(LlamacppConfig cfg)
    : impl_(std::make_unique<Impl>()), cfg_(std::move(cfg))
{
#if defined(LC_HAS_LLAMA)
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg_.n_gpu_layers;
    impl_->model.reset(llama_model_load_from_file(cfg_.model_path.c_str(), mparams));
    if (!impl_->model)
    {
        throw LCError("LlamacppLLM: failed to load " + cfg_.model_path);
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = cfg_.n_ctx;
    cparams.n_batch = cfg_.n_ctx;
    if (cfg_.n_threads > 0)
    {
        cparams.n_threads       = cfg_.n_threads;
        cparams.n_threads_batch = cfg_.n_threads;
    }
    impl_->ctx.reset(llama_init_from_model(impl_->model.get(), cparams));
    if (!impl_->ctx)
    {
        throw LCError("LlamacppLLM: failed to create context");
    }
    impl_->vocab = llama_model_get_vocab(impl_->model.get());
#else
    (void)cfg_;
#endif
}

LlamacppLLM::~LlamacppLLM() = default;

ChatResponse LlamacppLLM::invoke_impl(const ChatRequest& req)
{
#if defined(LC_HAS_LLAMA)
    std::uint32_t seed = cfg_.seed < 0
                             ? static_cast<std::uint32_t>(std::random_device{}())
                             : static_cast<std::uint32_t>(cfg_.seed);
    auto resp = generate(impl_->ctx.get(), impl_->vocab, impl_->model.get(),
                         impl_->tmpl_override, req, seed,
                         [](const std::string&) { return true; });
    resp.model = cfg_.model_path;
    return resp;
#else
    (void)req;
    ChatResponse out;
    out.model = cfg_.model_path;
    out.message = Message::assistant(
        "(llama backend not compiled in; rebuild with -DLC_ENABLE_LLAMA=ON)");
    out.finish_reason = "stop";
    return out;
#endif
}

ChatResponse LlamacppLLM::invoke_stream_impl(const ChatRequest& req,
                                              const StreamCallback& on_delta)
{
#if defined(LC_HAS_LLAMA)
    std::uint32_t seed = cfg_.seed < 0
                             ? static_cast<std::uint32_t>(std::random_device{}())
                             : static_cast<std::uint32_t>(cfg_.seed);
    auto resp = generate(impl_->ctx.get(), impl_->vocab, impl_->model.get(),
                         impl_->tmpl_override, req, seed,
                         [&](const std::string& piece) { return on_delta(piece); });
    resp.model = cfg_.model_path;
    return resp;
#else
    auto r = invoke_impl(req);
    on_delta(r.message.content);
    return r;
#endif
}

} // namespace llm
} // namespace langchain
