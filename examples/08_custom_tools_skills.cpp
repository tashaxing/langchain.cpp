// examples/08_custom_tools_skills.cpp -- demonstrate custom tool creation,
// SKILL.md-based skill loading, and agent composition.
//
// This example creates a custom C++ tool, loads skills from a directory
// of SKILL.md files, and gives both to an agent.
//
// Setup:
//   mkdir -p build/skills/calculator        # Linux/macOS
//   mkdir -p build_x64/skills/calculator    # Windows
//   cat > build/skills/calculator/SKILL.md <<'EOF'
//   # Calculator Skill
//
//   Evaluates arithmetic expressions.
//
//   ```json
//   {
//     "name": "calculator_skill",
//     "description": "Evaluate arithmetic expressions like (17 * 23) + 9",
//     "type": "prompt",
//     "template": "Evaluate the expression: {expression}\nResult: ",
//     "parameters": {
//       "type": "object",
//       "properties": {
//         "expression": {"type": "string", "description": "Math expression"}
//       },
//       "required": ["expression"]
//     }
//   }
//   ```
//   EOF
//
//   mkdir -p build/skills/greeter           # Linux/macOS
//   mkdir -p build_x64/skills/greeter       # Windows
//   cat > build/skills/greeter/SKILL.md <<'EOF'
//   # Greeter Skill
//
//   Generates a greeting message.
//
//   ```json
//   {
//     "name": "greeter",
//     "description": "Generate a greeting for a given name and language",
//     "type": "prompt",
//     "template": "Write a greeting for {name} in {language}.",
//     "parameters": {
//       "type": "object",
//       "properties": {
//         "name": {"type": "string"},
//         "language": {"type": "string"}
//       },
//       "required": ["name", "language"]
//     }
//   }
//   ```
//
// Run:
//   export LC_BASE_URL=http://localhost:11434
//   export LC_MODEL=qwen2.5:0.5b
//   ./build/08_custom_tools_skills              # Linux/macOS
//   ./build_x64/Debug/08_custom_tools_skills    # Windows
#include "langchain.h"
#include "skill/skill_loader.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

// A custom tool implemented in application code.
class ReverseTool : public langchain::tool::ITool
{
public:
    std::string name() const override
    {
        return "reverse";
    }

    std::string description() const override
    {
        return "Reverse the characters in a string.";
    }

    langchain::json parameters_schema() const override
    {
        return {
            {"type", "object"},
            {"properties", {
                {"text", {{"type", "string"}, {"description", "Text to reverse"}}}
            }},
            {"required", langchain::json::array({"text"})}
        };
    }

    std::string invoke(const langchain::json& args) override
    {
        std::string s = args.value("text", std::string());
        return std::string(s.rbegin(), s.rend());
    }
};

int main()
{
    using namespace langchain;

    llm::OpenAILLMConfig cfg;
    if (const char* e = std::getenv("LC_BASE_URL"))
    {
        cfg.base_url = e;
    }
    if (const char* e = std::getenv("LC_API_KEY"))
    {
        cfg.api_key = e;
    }
    if (const char* e = std::getenv("LC_MODEL"))
    {
        cfg.model = e;
    }
    auto llm = std::make_shared<llm::OpenAILLM>(cfg);

    // ---- Custom C++ tool ----
    tool::ToolRegistry tools;
    tools.add(std::make_shared<ReverseTool>());
    tools.add(tool::make_calculator_tool());

    // ---- Load skills from SKILL.md directory ----
    skill::SkillRegistry skills;
    try
    {
        std::filesystem::create_directories("build/skills");
        skill::SkillLoader::load_directory("build/skills", skills, llm);
        std::cout << "Loaded " << skills.size() << " skill(s):\n";
        for (const auto& n : skills.names())
        {
            std::cout << "  - " << n << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Skill loading failed: " << e.what() << "\n"
                  << "(Create build/skills/ with SKILL.md files to test.)\n";
    }

    // ---- Agent with both tools and skills ----
    agent::AgentConfig acfg;
    acfg.max_iterations = 6;
    acfg.system_prompt =
        "You are a helpful assistant with access to tools and skills.\n"
        "Use the calculator or calculator_skill for math.\n"
        "Use reverse to reverse text.\n"
        "Use greeter to generate greetings.";

    agent::ToolCallingAgent ag(llm, std::move(tools), std::move(skills), acfg);

    try
    {
        auto r = ag.invoke("What is (17 * 23) + 9? Also reverse the word 'hello'.");
        std::cout << "\nAnswer: " << r.output << "\n";
        std::cout << "Steps: " << r.steps.size()
                  << (r.finished ? " (finished)" : " (max-iter)") << "\n";
        for (const auto& s : r.steps)
        {
            std::cout << "  tool=" << s.tool_name
                      << " in=" << s.tool_input
                      << " out=" << s.observation << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "agent failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
