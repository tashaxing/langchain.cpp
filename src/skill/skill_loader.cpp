// src/skill/skill_loader.cpp -- SKILL.md discovery and parsing.
// Supports two formats:
//   1. YAML Frontmatter + Markdown Body (preferred)
//   2. Legacy: ```json code block inside markdown
#include "skill/skill_loader.h"

#include "util/fs.h"
#include "util/logging.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>

namespace langchain
{
namespace skill
{

namespace
{

// Extract YAML frontmatter from markdown text.
// Returns the frontmatter content (without --- delimiters) or empty string.
std::string extract_yaml_frontmatter(const std::string& markdown)
{
    if (markdown.size() < 4 || markdown[0] != '-' || markdown[1] != '-' || markdown[2] != '-')
    {
        return {};
    }

    std::size_t end = markdown.find("---", 3);
    if (end == std::string::npos)
    {
        return {};
    }

    std::string fm = markdown.substr(3, end - 3);
    // Trim leading/trailing whitespace/newlines.
    std::size_t first = 0;
    while (first < fm.size() && std::isspace(static_cast<unsigned char>(fm[first])))
    {
        ++first;
    }
    std::size_t last = fm.size();
    while (last > first && std::isspace(static_cast<unsigned char>(fm[last - 1])))
    {
        --last;
    }
    return fm.substr(first, last - first);
}

// Extract the first ```json ... ``` code block from markdown text.
// Returns empty string if none found.
std::string extract_json_block(const std::string& markdown)
{
    const std::string start_tag = "```json";
    const std::string end_tag   = "```";

    std::size_t start = markdown.find(start_tag);
    if (start == std::string::npos)
    {
        return {};
    }
    start += start_tag.size();

    // Skip newline after opening tag.
    if (start < markdown.size() && markdown[start] == '\n')
    {
        ++start;
    }
    else if (start + 1 < markdown.size() &&
             markdown[start] == '\r' && markdown[start + 1] == '\n')
    {
        start += 2;
    }

    std::size_t end = markdown.find(end_tag, start);
    if (end == std::string::npos)
    {
        return {};
    }

    // Trim trailing whitespace/newlines.
    while (end > start && std::isspace(static_cast<unsigned char>(markdown[end - 1])))
    {
        --end;
    }

    return markdown.substr(start, end - start);
}

// Convert a YAML::Node to nlohmann::json recursively.
json yaml_to_json(const YAML::Node& node)
{
    if (!node.IsDefined())
    {
        return json{};
    }

    switch (node.Type())
    {
    case YAML::NodeType::Null:
        return json(nullptr);

    case YAML::NodeType::Scalar:
    {
        std::string s = node.as<std::string>();
        // Try bool
        if (s == "true" || s == "True" || s == "TRUE")
            return json(true);
        if (s == "false" || s == "False" || s == "FALSE")
            return json(false);
        // Try integer
        try
        {
            std::size_t pos = 0;
            int iv = std::stoi(s, &pos);
            if (pos == s.size())
                return json(iv);
        }
        catch (const std::exception&) {}
        // Try float
        try
        {
            std::size_t pos = 0;
            double dv = std::stod(s, &pos);
            if (pos == s.size())
                return json(dv);
        }
        catch (const std::exception&) {}
        // Fallback to string
        return json(s);
    }

    case YAML::NodeType::Sequence:
    {
        json arr = json::array();
        for (const auto& item : node)
        {
            arr.push_back(yaml_to_json(item));
        }
        return arr;
    }

    case YAML::NodeType::Map:
    {
        json obj = json::object();
        for (const auto& kv : node)
        {
            obj[kv.first.as<std::string>()] = yaml_to_json(kv.second);
        }
        return obj;
    }

    default:
        return json{};
    }
}

void discover_recursive(const std::string& dir,
                        std::vector<std::string>& out)
{
    auto entries = util::fs::list_dir(dir);
    for (const auto& name : entries)
    {
        std::string full = util::fs::join(dir, name);
        if (util::fs::is_dir(full))
        {
            discover_recursive(full, out);
        }
        else if (name == "SKILL.md" || name == "skill.md")
        {
            out.push_back(full);
        }
    }
}

} // namespace

// ---------------- SkillLoader ----------------

std::vector<std::string> SkillLoader::discover(const std::string& dir_path)
{
    std::vector<std::string> out;
    if (!util::fs::is_dir(dir_path))
    {
        return out;
    }
    discover_recursive(dir_path, out);
    return out;
}

SkillManifest SkillLoader::parse_file(const std::string& path)
{
    std::string content = util::fs::read_file(path);

    json j;
    bool is_yaml_frontmatter = false;

    // Try YAML frontmatter first.
    std::string yaml_fm = extract_yaml_frontmatter(content);
    if (!yaml_fm.empty())
    {
        try
        {
            YAML::Node root = YAML::Load(yaml_fm);
            j = yaml_to_json(root);
            is_yaml_frontmatter = true;
        }
        catch (const std::exception& e)
        {
            throw LCError(std::string("SKILL.md YAML frontmatter parse error: ") + e.what() + ": " + path);
        }
    }
    else
    {
        // Fallback to legacy ```json block.
        std::string json_text = extract_json_block(content);
        if (json_text.empty())
        {
            throw LCError("SKILL.md has no YAML frontmatter or ```json block: " + path);
        }

        j = json::parse(json_text, nullptr, false);
        if (j.is_discarded())
        {
            throw LCError("SKILL.md JSON block is invalid: " + path);
        }
    }

    SkillManifest m;
    m.name        = j.value("name", std::string());
    m.description = j.value("description", std::string());
    m.type        = j.value("type", std::string("prompt"));
    m.template_str = j.value("template", std::string());
    m.parameters  = j.value("parameters", json::object());

    // For YAML frontmatter, also merge any extra keys into parameters so that
    // script_dir, script_name, etc. are accessible.
    if (is_yaml_frontmatter && j.is_object())
    {
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            const std::string& key = it.key();
            if (key != "name" && key != "description" && key != "type" &&
                key != "template" && key != "parameters")
            {
                m.parameters[key] = it.value();
            }
        }
    }

    if (j.contains("tags") && j["tags"].is_array())
    {
        for (const auto& t : j["tags"])
        {
            if (t.is_string())
            {
                m.tags.push_back(t.get<std::string>());
            }
        }
    }

    if (m.name.empty())
    {
        throw LCError("SKILL.md missing 'name' field: " + path);
    }
    if (m.template_str.empty())
    {
        throw LCError("SKILL.md missing 'template' field: " + path);
    }

    return m;
}

SkillPtr SkillLoader::load(const SkillManifest& manifest,
                           llm::LLMPtr llm,
                           vectorstore::VectorStorePtr vs)
{
    if (manifest.type != "script" && !llm)
    {
        throw LCError("SkillLoader::load requires a non-null LLM");
    }

    prompt::PromptTemplate tmpl(manifest.template_str);

    if (manifest.type == "retrieval")
    {
        if (!vs)
        {
            throw LCError("SkillLoader::load: 'retrieval' skill requires a vector store");
        }
        return std::make_shared<RetrievalSkill>(
            manifest.name,
            manifest.description,
            std::move(llm),
            std::move(vs),
            std::move(tmpl));
    }

    if (manifest.type == "script")
    {
        // Script skills do not require an LLM.
        // The script_path is resolved by the application layer which knows
        // the skill directory path. We pass an empty string here and let
        // the app set the correct path after loading.
        return std::make_shared<ScriptSkill>(
            manifest.name,
            manifest.description,
            std::string() /* resolved by app layer */);
    }

    // Default to prompt skill.
    return std::make_shared<PromptSkill>(
        manifest.name,
        manifest.description,
        std::move(llm),
        std::move(tmpl));
}

void SkillLoader::load_directory(const std::string& dir_path,
                                 SkillRegistry& registry,
                                 llm::LLMPtr llm,
                                 vectorstore::VectorStorePtr vs)
{
    auto files = discover(dir_path);
    for (const auto& path : files)
    {
        try
        {
            auto manifest = parse_file(path);
            auto skill = load(manifest, llm, vs);
            if (skill)
            {
                // For script skills, resolve the script path relative to the
                // skill directory (the directory containing SKILL.md).
                if (manifest.type == "script")
                {
                    auto* script_skill = dynamic_cast<ScriptSkill*>(skill.get());
                    if (script_skill)
                    {
                        std::string script_dir = manifest.parameters.value("script_dir", std::string("scripts"));
                        std::string script_name = manifest.parameters.value("script_name", manifest.name);
                        std::string skill_dir = util::fs::parent(path);
                        std::string script_path = util::fs::join(skill_dir, script_dir);
                        script_path = util::fs::join(script_path, script_name);
                        // Re-create with resolved path.
                        skill = std::make_shared<ScriptSkill>(
                            manifest.name,
                            manifest.description,
                            std::move(script_path));
                    }
                }
                registry.add(std::move(skill));
            }
        }
        catch (const std::exception& e)
        {
            LOG_WARN("[SkillLoader] failed to load '{}': {}", path, e.what());
        }
    }
}

} // namespace skill
} // namespace langchain
