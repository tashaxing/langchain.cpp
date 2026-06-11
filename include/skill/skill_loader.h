// langchain/skill/skill_loader.h
// Load skills from a directory tree that follows the SKILL.md convention.
//
// Directory layout:
//   skills/
//   ├── calculator/
//   │   └── SKILL.md
//   └── web_search/
//       └── SKILL.md
//
// Each SKILL.md is a Markdown file that contains a JSON code block with
// skill metadata. Supported skill types:
//   - "prompt"      -- PromptSkill (requires "template" field)
//   - "retrieval"   -- RetrievalSkill (requires "template" + vector store)
//
// Example SKILL.md:
//   # Calculator
//
//   A skill that evaluates arithmetic expressions.
//
//   ```json
//   {
//     "name": "calculator",
//     "description": "Evaluate arithmetic expressions",
//     "type": "prompt",
//     "template": "Evaluate: {expression}",
//     "parameters": {
//       "type": "object",
//       "properties": {
//         "expression": {"type": "string", "description": "Math expression"}
//       },
//       "required": ["expression"]
//     }
//   }
//   ```
#pragma once

#include "skill/skill.h"
#include "skill/skill_registry.h"
#include "vectorstore/vectorstore.h"

#include <string>
#include <vector>

namespace langchain
{
namespace skill
{

// Parsed metadata from a SKILL.md file.
struct SkillManifest
{
    std::string name;
    std::string description;
    std::string type;        // "prompt" or "retrieval"
    std::string template_str;
    json parameters;         // JSON schema for tool export
    std::vector<std::string> tags;
};

class SkillLoader
{
public:
    // Discover all SKILL.md files under `dir_path` (recursive).
    // Returns full paths to each SKILL.md found.
    static std::vector<std::string> discover(const std::string& dir_path);

    // Parse a single SKILL.md file into a manifest.
    // Throws LCError on parse failure.
    static SkillManifest parse_file(const std::string& path);

    // Load a skill from a manifest. `llm` is required; `vs` is only used
    // for "retrieval" type skills.
    static SkillPtr load(const SkillManifest& manifest,
                         llm::LLMPtr llm,
                         vectorstore::VectorStorePtr vs = nullptr);

    // Convenience: scan a directory, parse every SKILL.md, and register
    // the resulting skills into a SkillRegistry.
    static void load_directory(const std::string& dir_path,
                               SkillRegistry& registry,
                               llm::LLMPtr llm,
                               vectorstore::VectorStorePtr vs = nullptr);
};

} // namespace skill
} // namespace langchain
