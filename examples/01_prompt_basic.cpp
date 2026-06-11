// examples/01_prompt_basic.cpp — PromptTemplate round-trip without any network.
#include "langchain.h"

#include <iostream>

int main()
{
    using namespace langchain;

    prompt::PromptTemplate t("Hello {name}, today's topic is {topic}.");
    std::cout << "Variables: ";
    for (const auto& v : t.input_variables())
    {
        std::cout << v << " ";
    }
    std::cout << "\n";

    std::cout << t.format({{"name", "Alice"}, {"topic", "C++ modules"}}) << "\n";
    return 0;
}
