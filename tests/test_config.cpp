// tests/test_config.cpp
// GoogleTest suite for Config / ConfigSection.

#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

#include <gtest/gtest.h>

#include "util/config.h"
#include "util/singleton.h"

namespace fs = std::filesystem;
using langchain::util::Config;
using langchain::util::ConfigSection;
using langchain::util::Singleton;

class ConfigTest : public ::testing::Test
{
protected:
    std::string tmp_dir_;
    std::string test_file_;

    void SetUp() override
    {
        tmp_dir_ = (fs::temp_directory_path() / "lc_config_test").string();
        fs::create_directories(tmp_dir_);
        test_file_ = (fs::path(tmp_dir_) / "test.xml").string();

        // Reset singleton state by reloading an empty valid XML.
        write_xml(R"(<?xml version="1.0"?><config></config>)");
        auto& cfg = Singleton<Config>::instance();
        cfg.load(test_file_);
    }

    void TearDown() override
    {
        fs::remove_all(tmp_dir_);
    }

    void write_xml(const std::string& content)
    {
        std::ofstream ofs(test_file_, std::ios::binary);
        ofs << content;
    }
};

TEST_F(ConfigTest, LoadAndGetString)
{
    write_xml(R"(<?xml version="1.0"?>
<config>
  <app>
    <name>demo</name>
    <enabled>true</enabled>
  </app>
</config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    const auto& app = cfg.section("app");
    EXPECT_EQ(app.get("name", std::string{}), "demo");
    EXPECT_EQ(app.get("enabled", std::string{}), "true");
}

TEST_F(ConfigTest, GetTypedValues)
{
    write_xml(R"(<?xml version="1.0"?>
<config>
  <server>
    <port>8080</port>
    <timeout>3.5</timeout>
    <active>true</active>
  </server>
</config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    const auto& srv = cfg.section("server");
    EXPECT_EQ(srv.get<int>("port", 0), 8080);
    EXPECT_DOUBLE_EQ(srv.get<double>("timeout", 0.0), 3.5);
    EXPECT_EQ(srv.get<std::string>("active", ""), "true");
}

TEST_F(ConfigTest, MissingKeyReturnsDefault)
{
    write_xml(R"(<?xml version="1.0"?><config><app></app></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    const auto& app = cfg.section("app");
    EXPECT_EQ(app.get("missing", std::string("fallback")), "fallback");
    EXPECT_EQ(app.get<int>("missing", 42), 42);
}

TEST_F(ConfigTest, MissingSectionThrows)
{
    write_xml(R"(<?xml version="1.0"?><config></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    const Config& ccfg = cfg;
    EXPECT_THROW(ccfg.section("nope"), std::out_of_range);
    EXPECT_FALSE(cfg.has_section("nope"));
}

TEST_F(ConfigTest, SetAndSave)
{
    write_xml(R"(<?xml version="1.0"?><config><app><name>x</name></app></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    cfg.section("app").set("name", "updated");
    cfg.section("app").set("count", 7);

    ASSERT_TRUE(cfg.save());

    // Reload and verify persistence.
    ASSERT_TRUE(cfg.reload());
    const auto& app = cfg.section("app");
    EXPECT_EQ(app.get("name", std::string{}), "updated");
    EXPECT_EQ(app.get<int>("count", 0), 7);
}

TEST_F(ConfigTest, SaveAsUpdatesPath)
{
    write_xml(R"(<?xml version="1.0"?><config><a><k>v</k></a></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    std::string other = (fs::path(tmp_dir_) / "other.xml").string();
    ASSERT_TRUE(cfg.save_as(other));
    EXPECT_EQ(cfg.file_path(), other);

    ASSERT_TRUE(cfg.reload());
    EXPECT_EQ(cfg.section("a").get("k", std::string{}), "v");
}

TEST_F(ConfigTest, CheckReloadDetectsChange)
{
    write_xml(R"(<?xml version="1.0"?><config><app><v>1</v></app></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    // Ensure filesystem timestamp granularity (1s on some platforms).
    std::this_thread::sleep_for(std::chrono::seconds(1));

    write_xml(R"(<?xml version="1.0"?><config><app><v>2</v></app></config>)");

    EXPECT_TRUE(cfg.check_reload());
    EXPECT_EQ(cfg.section("app").get("v", std::string{}), "2");

    // No change now.
    EXPECT_FALSE(cfg.check_reload());
}

TEST_F(ConfigTest, RemoveSection)
{
    write_xml(R"(<?xml version="1.0"?><config><a></a><b></b></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    cfg.remove_section("a");
    EXPECT_FALSE(cfg.has_section("a"));
    EXPECT_TRUE(cfg.has_section("b"));
}

TEST_F(ConfigTest, SectionNames)
{
    write_xml(R"(<?xml version="1.0"?><config><z></z><a></a></config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    auto names = cfg.section_names();
    ASSERT_EQ(names.size(), 2u);
    // Order follows unordered_map iteration; just check presence.
    EXPECT_TRUE(std::find(names.begin(), names.end(), "a") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "z") != names.end());
}

TEST_F(ConfigTest, ConfigSectionSnapshot)
{
    ConfigSection sec("demo");
    sec.set("x", 1);
    sec.set("y", "hello");

    auto snap = sec.snapshot();
    EXPECT_EQ(snap.size(), 2u);
    EXPECT_EQ(snap["x"], "1");
    EXPECT_EQ(snap["y"], "hello");
}

TEST_F(ConfigTest, ConfigSectionHasAndRemove)
{
    ConfigSection sec("demo");
    sec.set("k", "v");
    EXPECT_TRUE(sec.has("k"));
    sec.remove("k");
    EXPECT_FALSE(sec.has("k"));
}

TEST_F(ConfigTest, EnvVarInterpolation)
{
#ifdef _WIN32
    ::_putenv_s("LC_TEST_VAR", "interpolated_value");
#else
    ::setenv("LC_TEST_VAR", "interpolated_value", 1);
#endif

    write_xml(R"(<?xml version="1.0"?>
<config>
  <app>
    <name>${LC_TEST_VAR}</name>
    <desc>prefix-${LC_TEST_VAR}-suffix</desc>
    <missing>${LC_TEST_MISSING:-default}</missing>
  </app>
</config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    const auto& app = cfg.section("app");
    EXPECT_EQ(app.get("name", std::string{}), "interpolated_value");
    EXPECT_EQ(app.get("desc", std::string{}), "prefix-interpolated_value-suffix");
    EXPECT_EQ(app.get("missing", std::string{}), "default");

#ifdef _WIN32
    ::_putenv_s("LC_TEST_VAR", "");
#else
    ::unsetenv("LC_TEST_VAR");
#endif
}

TEST_F(ConfigTest, Validation)
{
    write_xml(R"(<?xml version="1.0"?>
<config>
  <app>
    <name>demo</name>
    <port>8080</port>
  </app>
</config>)");

    auto& cfg = Singleton<Config>::instance();
    ASSERT_TRUE(cfg.load(test_file_));

    // All rules pass.
    auto errors = cfg.validate({
        {"app", "name", true, nullptr},
        {"app", "port", true, [](const std::string& v) {
            return !v.empty() && std::all_of(v.begin(), v.end(), ::isdigit);
        }},
    });
    EXPECT_TRUE(errors.empty());

    // Missing required key.
    errors = cfg.validate({
        {"app", "missing_key", true, nullptr},
    });
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].find("missing required key 'missing_key'"), std::string::npos);

    // Missing required section.
    errors = cfg.validate({
        {"nonexistent", "key", true, nullptr},
    });
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].find("missing required section"), std::string::npos);

    // Value check fails.
    errors = cfg.validate({
        {"app", "name", true, [](const std::string& v) {
            return v == "expected";
        }},
    });
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].find("invalid value"), std::string::npos);

    // Optional missing key is OK.
    errors = cfg.validate({
        {"app", "optional_key", false, nullptr},
    });
    EXPECT_TRUE(errors.empty());
}
