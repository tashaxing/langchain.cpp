cmake_minimum_required(VERSION 3.15)
project(langchain_cpp C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/utf-8 /W3 /EHsc /Zc:__cplusplus /permissive-)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# ---- threads (pthread on Linux/Mac) ----
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---- nlohmann_json (header-only) ----
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    $<BUILD_INTERFACE:${DEPS_DIR}/json-v3.12.0/include>)

# ---- cpp-httplib (header-only) ----
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE
    $<BUILD_INTERFACE:${DEPS_DIR}/cpp-httplib-v0.46.0>)
target_link_libraries(httplib INTERFACE Threads::Threads)
if(WIN32)
    target_link_libraries(httplib INTERFACE ws2_32 crypt32)
endif()

# Optional HTTPS support. Off by default because it triggers a full bundled
# OpenSSL build (perl + nmake/make required). Turn on for HTTPS endpoints
# like api.openai.com.
option(LC_ENABLE_OPENSSL "Enable HTTPS support; builds bundled OpenSSL from deps/openssl-openssl-3.3.6" OFF)
if(LC_ENABLE_OPENSSL)
    include(ExternalProject)

    set(OPENSSL_SRC      ${DEPS_DIR}/openssl-openssl-3.3.6)
    set(OPENSSL_PREFIX   ${CMAKE_BINARY_DIR}/openssl_build)
    set(OPENSSL_INSTALL  ${OPENSSL_PREFIX}/install)

    if(WIN32)
        # MSVC: VC-WIN64A target, nmake. Requires perl in PATH. `no-asm`
        # avoids the NASM dependency at a slight perf cost (acceptable for
        # a build-from-source convenience target).
        set(_ossl_target VC-WIN64A)
        set(_ossl_libssl    ${OPENSSL_INSTALL}/lib/libssl.lib)
        set(_ossl_libcrypto ${OPENSSL_INSTALL}/lib/libcrypto.lib)
        set(_ossl_configure_cmd
            perl ${OPENSSL_SRC}/Configure ${_ossl_target}
                 no-shared no-asm no-tests no-docs no-apps
                 --prefix=${OPENSSL_INSTALL} --openssldir=${OPENSSL_INSTALL}/ssl)
        set(_ossl_build_cmd   nmake)
        set(_ossl_install_cmd nmake install_sw)
    elseif(APPLE)
        set(_ossl_target darwin64-x86_64-cc)
        set(_ossl_libssl    ${OPENSSL_INSTALL}/lib/libssl.a)
        set(_ossl_libcrypto ${OPENSSL_INSTALL}/lib/libcrypto.a)
        set(_ossl_configure_cmd
            perl ${OPENSSL_SRC}/Configure ${_ossl_target}
                 no-shared no-tests no-docs no-apps
                 --prefix=${OPENSSL_INSTALL} --openssldir=${OPENSSL_INSTALL}/ssl)
        set(_ossl_build_cmd   make -j)
        set(_ossl_install_cmd make install_sw)
    else()
        set(_ossl_target linux-x86_64)
        set(_ossl_libssl    ${OPENSSL_INSTALL}/lib64/libssl.a)
        set(_ossl_libcrypto ${OPENSSL_INSTALL}/lib64/libcrypto.a)
        # OpenSSL on some distros installs to lib/ instead of lib64/; fall back.
        set(_ossl_configure_cmd
            perl ${OPENSSL_SRC}/Configure ${_ossl_target}
                 no-shared no-tests no-docs no-apps
                 --prefix=${OPENSSL_INSTALL} --openssldir=${OPENSSL_INSTALL}/ssl
                 --libdir=lib64)
        set(_ossl_build_cmd   make -j)
        set(_ossl_install_cmd make install_sw)
    endif()

    ExternalProject_Add(openssl_ext
        SOURCE_DIR        ${OPENSSL_SRC}
        PREFIX            ${OPENSSL_PREFIX}
        CONFIGURE_COMMAND ${_ossl_configure_cmd}
        BUILD_COMMAND     ${_ossl_build_cmd}
        INSTALL_COMMAND   ${_ossl_install_cmd}
        BUILD_IN_SOURCE   1
        BUILD_BYPRODUCTS  ${_ossl_libssl} ${_ossl_libcrypto}
        LOG_CONFIGURE     1
        LOG_BUILD         1
        LOG_INSTALL       1)

    file(MAKE_DIRECTORY ${OPENSSL_INSTALL}/include)

    add_library(openssl_crypto STATIC IMPORTED)
    set_target_properties(openssl_crypto PROPERTIES
        IMPORTED_LOCATION             ${_ossl_libcrypto}
        INTERFACE_INCLUDE_DIRECTORIES ${OPENSSL_INSTALL}/include)
    add_dependencies(openssl_crypto openssl_ext)

    add_library(openssl_ssl STATIC IMPORTED)
    set_target_properties(openssl_ssl PROPERTIES
        IMPORTED_LOCATION             ${_ossl_libssl}
        INTERFACE_INCLUDE_DIRECTORIES ${OPENSSL_INSTALL}/include
        INTERFACE_LINK_LIBRARIES      openssl_crypto)
    add_dependencies(openssl_ssl openssl_ext)

    target_compile_definitions(httplib INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(httplib INTERFACE openssl_ssl openssl_crypto)
    if(WIN32)
        target_link_libraries(httplib INTERFACE crypt32 ws2_32 advapi32 user32)
    elseif(NOT APPLE)
        target_link_libraries(httplib INTERFACE ${CMAKE_DL_LIBS})
    endif()
endif()

# ---- spdlog (header-only mode) ----
add_library(spdlog INTERFACE)
target_include_directories(spdlog INTERFACE
    $<BUILD_INTERFACE:${DEPS_DIR}/spdlog-v1.13.0/include>)
target_compile_definitions(spdlog INTERFACE FMT_HEADER_ONLY)

# ---- zlib (bundled, built from source) ----
# Build it as a subproject. We disable shared / install / tests to keep
# the third-party footprint minimal and to avoid polluting our install.
set(ZLIB_BUILD_SHARED  OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC  ON  CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL       OFF CACHE BOOL "" FORCE)
add_subdirectory(${DEPS_DIR}/zlib-v1.3.2 ${CMAKE_BINARY_DIR}/zlib_build EXCLUDE_FROM_ALL)

# ---- yaml-cpp ----
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${DEPS_DIR}/yaml-cpp-yaml-cpp-0.9.0 ${CMAKE_BINARY_DIR}/yamlcpp_build EXCLUDE_FROM_ALL)

# ---- pugixml ----
add_library(pugixml STATIC ${DEPS_DIR}/pugixml-v1.15/src/pugixml.cpp)
target_include_directories(pugixml PUBLIC
    $<BUILD_INTERFACE:${DEPS_DIR}/pugixml-v1.15/src>)

# ---- sqlite3 ----
set(SQLITE_C ${DEPS_DIR}/sqlite-3.51/sqlite3.c)
if(EXISTS ${SQLITE_C})
    add_library(sqlite3 STATIC ${SQLITE_C})
    set_source_files_properties(${SQLITE_C} PROPERTIES LANGUAGE C)
    target_include_directories(sqlite3 PUBLIC
        $<BUILD_INTERFACE:${DEPS_DIR}/sqlite-3.51>)
    if(NOT WIN32)
        # sqlite needs dl + pthread on Linux; -lm for math.
        target_link_libraries(sqlite3 PUBLIC ${CMAKE_DL_LIBS} Threads::Threads m)
    endif()
else()
    message(WARNING "sqlite3.c not found under ${DEPS_DIR}/sqlite-3.51 -- vectorstore SQLite backend will be stubbed.")
    add_library(sqlite3 INTERFACE)
endif()

# ---- llama.cpp (optional, heavy) ----
option(LC_ENABLE_LLAMA "Enable bundled llama.cpp backend" OFF)
if(LC_ENABLE_LLAMA AND EXISTS ${DEPS_DIR}/llama.cpp-b5018/CMakeLists.txt)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    add_subdirectory(${DEPS_DIR}/llama.cpp-b5018 ${CMAKE_BINARY_DIR}/llama_build EXCLUDE_FROM_ALL)
endif()

# ---- faiss (optional) ----
option(LC_ENABLE_FAISS "Enable FAISS-backed vector store" OFF)
if(LC_ENABLE_FAISS AND EXISTS ${DEPS_DIR}/faiss-v1.14.2/CMakeLists.txt)
    # Build faiss as a subproject.
    set(FAISS_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(FAISS_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
    set(FAISS_BUILD_SHARED     OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING          OFF CACHE BOOL "" FORCE)
    add_subdirectory(${DEPS_DIR}/faiss-v1.14.2 ${CMAKE_BINARY_DIR}/faiss_build EXCLUDE_FROM_ALL)
endif()

# ---- langchain library ----
file(GLOB_RECURSE LC_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

if(NOT LC_ENABLE_LLAMA)
    list(FILTER LC_SOURCES EXCLUDE REGEX ".*llamacpp_llm\\.cpp$")
    list(FILTER LC_SOURCES EXCLUDE REGEX ".*llamacpp_embedding\\.cpp$")
endif()

add_library(langchain STATIC ${LC_SOURCES})
target_include_directories(langchain PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(langchain
    PUBLIC nlohmann_json httplib spdlog pugixml sqlite3 zlibstatic Threads::Threads
    PRIVATE yaml-cpp)

if(MSVC)
    # src/api/server.cpp pulls in httplib + nlohmann_json and contains most of
    # the OpenAI-compatible route machinery. MSVC can run out of compiler heap
    # on this large TU, especially in Debug/Release after route-handling grows.
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/api/server.cpp
        PROPERTIES COMPILE_OPTIONS "/bigobj;/Zm200")
endif()

if(LC_ENABLE_FAISS AND TARGET faiss)
    target_link_libraries(langchain PUBLIC faiss)
    target_compile_definitions(langchain PUBLIC LC_HAS_FAISS=1)
endif()

# GCC < 9 needs explicit libstdc++fs for std::filesystem (we don't use it
# now, but link it pre-emptively for any future code).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(langchain PUBLIC stdc++fs)
endif()

if(LC_ENABLE_LLAMA AND TARGET llama)
    target_link_libraries(langchain PUBLIC llama)
    target_compile_definitions(langchain PUBLIC LC_HAS_LLAMA=1)
endif()

# ---- FLTK (optional, desktop UI) ----
option(LC_BUILD_FLTK_UI "Build FLTK desktop UI examples" OFF)
if(LC_BUILD_FLTK_UI AND EXISTS ${DEPS_DIR}/fltk-release-1.4.5/CMakeLists.txt)
    set(FLTK_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_TEST        OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_FLUID       OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_FLTK_OPTIONS OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_FORMS       OFF CACHE BOOL "" FORCE)
    set(FLTK_BUILD_GL          OFF CACHE BOOL "" FORCE)
    set(FLTK_GRAPHICS_GDIPLUS  OFF CACHE BOOL "" FORCE)
    add_subdirectory(${DEPS_DIR}/fltk-release-1.4.5 ${CMAKE_BINARY_DIR}/fltk_build EXCLUDE_FROM_ALL)
endif()

# ---- examples ----
option(LC_BUILD_EXAMPLES "Build langchain.cpp examples" ON)
if(LC_BUILD_EXAMPLES)
    # Set output directory for all example executables.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

    # 00_smart_app has its own CMakeLists.txt (multi-file target).
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/00_smart_app/CMakeLists.txt)
        add_subdirectory(examples/00_smart_app)
    endif()

    # All other examples are single .cpp files.
    file(GLOB LC_EXAMPLES CONFIGURE_DEPENDS examples/*.cpp)
    foreach(ex ${LC_EXAMPLES})
        get_filename_component(name ${ex} NAME_WE)
        # Skip the 00_smart_app placeholder if it ever gets a .cpp at top level.
        if(name STREQUAL "00_smart_app")
            continue()
        endif()
        add_executable(${name} ${ex})
        target_link_libraries(${name} PRIVATE langchain)
        # Ensure execute permission on Unix (CMake <3.19 lacks -E chmod).
    endforeach()
endif()

# ---- tests ----
option(LC_BUILD_TESTS "Build langchain.cpp tests" ON)
if(LC_BUILD_TESTS)
    enable_testing()

    # Set output directory for all test executables.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # GoogleTest-based tests (bundled). On Windows, force shared CRT so
    # gtest matches our other libs (/MD vs /MT).
    if(EXISTS ${DEPS_DIR}/googletest-v1.17.0/CMakeLists.txt)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK              OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST            OFF CACHE BOOL "" FORCE)
        add_subdirectory(${DEPS_DIR}/googletest-v1.17.0
                         ${CMAKE_BINARY_DIR}/gtest_build
                         EXCLUDE_FROM_ALL)

        file(GLOB LC_GTESTS CONFIGURE_DEPENDS tests/test_*.cpp)
        if(NOT LC_ENABLE_LLAMA)
            list(FILTER LC_GTESTS EXCLUDE REGEX ".*test_llamacpp_embedding\\.cpp$")
        endif()
        foreach(tf ${LC_GTESTS})
            get_filename_component(name ${tf} NAME_WE)
            add_executable(${name} ${tf})
            target_link_libraries(${name} PRIVATE langchain gtest gtest_main)
            add_test(NAME ${name} COMMAND ${CMAKE_BINARY_DIR}/tests/${name})
        endforeach()
    endif()
endif()

# ---------------------------------------------------------------------------
# Install / packaging — staged into the build tree only; never writes to
# system directories like /usr/local.  Downstream projects can consume
# langchain.cpp via add_subdirectory() or by pointing to the staged tree.
# ---------------------------------------------------------------------------
option(LC_INSTALL "Stage installable artifacts into the build tree" ON)
if(LC_INSTALL)
    set(LC_STAGE_DIR ${CMAKE_BINARY_DIR}/stage)

    # Headers: preserve include/ layout verbatim.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
            DESTINATION ${LC_STAGE_DIR}/include
            FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp")

    # Vendored third-party headers that leak through our public API.
    install(DIRECTORY ${DEPS_DIR}/json-v3.12.0/include/
            DESTINATION ${LC_STAGE_DIR}/include
            FILES_MATCHING PATTERN "*.hpp")
    install(FILES ${DEPS_DIR}/cpp-httplib-v0.46.0/httplib.h
            DESTINATION ${LC_STAGE_DIR}/include)
    install(DIRECTORY ${DEPS_DIR}/spdlog-v1.13.0/include/spdlog
            DESTINATION ${LC_STAGE_DIR}/include
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
    install(DIRECTORY ${DEPS_DIR}/pugixml-v1.15/src/
            DESTINATION ${LC_STAGE_DIR}/include
            FILES_MATCHING PATTERN "pugixml.hpp" PATTERN "pugiconfig.hpp")
    if(EXISTS ${DEPS_DIR}/sqlite-3.51/sqlite3.h)
        install(FILES ${DEPS_DIR}/sqlite-3.51/sqlite3.h
                DESTINATION ${LC_STAGE_DIR}/include)
    endif()
    install(FILES ${DEPS_DIR}/zlib-v1.3.2/zlib.h
            DESTINATION ${LC_STAGE_DIR}/include)
    if(EXISTS ${CMAKE_BINARY_DIR}/zlib_build/zconf.h)
        install(FILES ${CMAKE_BINARY_DIR}/zlib_build/zconf.h
                DESTINATION ${LC_STAGE_DIR}/include)
    endif()

    # Static libraries.
    install(TARGETS langchain nlohmann_json httplib spdlog yaml-cpp pugixml sqlite3 zlibstatic
            ARCHIVE DESTINATION ${LC_STAGE_DIR}/lib
            LIBRARY DESTINATION ${LC_STAGE_DIR}/lib
            RUNTIME DESTINATION ${LC_STAGE_DIR}/bin
            INCLUDES DESTINATION ${LC_STAGE_DIR}/include)
endif()

# Temporary debug target
# add_executable(test_config_debug test_config.cpp)
# target_link_libraries(test_config_debug PRIVATE langchain)
# add_executable(test_config2 test_config2.cpp)
# target_link_libraries(test_config2 PRIVATE langchain)
# add_executable(test_config3 test_config3.cpp)
# target_link_libraries(test_config3 PRIVATE langchain)
# add_executable(test_gtest_rag_debug test_gtest_rag_debug.cpp)
# target_link_libraries(test_gtest_rag_debug PRIVATE langchain gtest gtest_main)
# add_executable(test_config_odr test_config_odr.cpp)
# target_link_libraries(test_config_odr PRIVATE langchain)
