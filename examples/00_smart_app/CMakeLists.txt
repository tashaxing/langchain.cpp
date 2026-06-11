# examples/00_smart_app/CMakeLists.txt

set(SMART_APP_SOURCES
    main.cpp
    app_config.cpp
    app_paths.cpp
    app_tools.cpp
    app_hooks.cpp
    app_rag.cpp
    app_agent.cpp
    app_server.cpp
)

add_executable(00_smart_app ${SMART_APP_SOURCES})

target_link_libraries(00_smart_app PRIVATE langchain)

# Override the global runtime output directory so the binary lands in
# examples/00_smart_app/bin/ directly during the build.
# For multi-config generators (Visual Studio) we also pin the per-config
# output so that Debug/Release subdirectories are NOT created.
set_target_properties(00_smart_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(SMART_APP_COPY_SCRIPTS_CMAKE ${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_scripts.cmake)
file(WRITE ${SMART_APP_COPY_SCRIPTS_CMAKE}
"file(COPY \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/start.sh\"
          \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/stop.sh\"
     DESTINATION \"${CMAKE_CURRENT_BINARY_DIR}/bin\"
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE)
file(COPY \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/start.bat\"
          \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/stop.bat\"
     DESTINATION \"${CMAKE_CURRENT_BINARY_DIR}/bin\")
")

# ---------------------------------------------------------------------------
# Build-time: mirror the deployment directory structure in the build dir
# so the executable can find assets at runtime during development.
#
#   build/examples/00_smart_app/
#   ├── bin/00_smart_app
#   ├── bin/start.sh
#   ├── bin/stop.sh
#   ├── bin/start.bat
#   ├── bin/stop.bat
#   ├── config/
#   ├── data/
#   ├── log/          (created at runtime)
#   ├── lib/          (optional, for shared libs)
#   └── skills/
#
# Note: the executable is placed in bin/ only. The root directory does
# not contain the binary (it is not a valid runtime location).
# ---------------------------------------------------------------------------
add_custom_command(TARGET 00_smart_app POST_BUILD
    # Create directory structure
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/log
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/lib

    # Copy runtime assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/config
        ${CMAKE_CURRENT_BINARY_DIR}/config
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/skills
        ${CMAKE_CURRENT_BINARY_DIR}/skills
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data

    # Executable is already in bin/ via RUNTIME_OUTPUT_DIRECTORY

    # Copy start/stop scripts into bin/ with executable bits for Unix scripts.
    COMMAND ${CMAKE_COMMAND} -P ${SMART_APP_COPY_SCRIPTS_CMAKE}

    COMMENT "Setting up 00_smart_app deployment directory structure"
)

# ---------------------------------------------------------------------------
# Install rules: deploy to a self-contained directory.
# After `cmake --install`, the layout is:
#
#   ${CMAKE_INSTALL_PREFIX}/
#   ├── bin/
#   │   ├── 00_smart_app
#   │   ├── start.sh
#   │   ├── stop.sh
#   │   ├── start.bat
#   │   └── stop.bat
#   ├── config/
#   │   └── app_config.xml
#   ├── data/
#   │   └── knowledge/
#   ├── log/          (created at runtime)
#   ├── lib/          (optional)
#   └── skills/
# ---------------------------------------------------------------------------
# ---- FLTK desktop chat UI (optional) ----
if(LC_BUILD_FLTK_UI AND TARGET fltk::fltk)
    add_subdirectory(ui)
endif()

if(LC_INSTALL)
    # Executable
    install(TARGETS 00_smart_app
            RUNTIME DESTINATION bin)

    # Config files
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config/
            DESTINATION config
            FILES_MATCHING PATTERN "*.xml" PATTERN "*.yaml" PATTERN "*.yml" PATTERN "*.json")

    # Data files
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
            DESTINATION data)

    # Skills
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/skills/
            DESTINATION skills
            USE_SOURCE_PERMISSIONS)

    # Scripts (start/stop)
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/start.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/stop.sh
            DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/start.bat
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/stop.bat
            DESTINATION bin)

    # Create empty log/ and lib/ directories at install time
    install(DIRECTORY DESTINATION log)
    install(DIRECTORY DESTINATION lib)
endif()
