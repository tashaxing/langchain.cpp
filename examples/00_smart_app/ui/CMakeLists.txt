# examples/00_smart_app/ui/CMakeLists.txt

set(MD4C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../deps/md4c-release-0.5.3/src)

add_library(smart_app_fltk_ui STATIC
    chat_app.cpp
    chat_bubble_view.cpp
    chat_config.cpp
    smart_app_client.cpp
    ${MD4C_SRC}/md4c.c
)

target_include_directories(smart_app_fltk_ui
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${MD4C_SRC})

target_link_libraries(smart_app_fltk_ui
    PUBLIC langchain fltk::fltk)

add_executable(00_smart_app_chat
    main_fltk_chat.cpp
)

target_link_libraries(00_smart_app_chat
    PRIVATE smart_app_fltk_ui)

set_target_properties(00_smart_app_chat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/../bin)

if(WIN32)
    set_target_properties(00_smart_app_chat PROPERTIES WIN32_EXECUTABLE TRUE)
    if(MSVC)
        target_link_options(00_smart_app_chat PRIVATE "/ENTRY:mainCRTStartup")
    endif()
endif()

add_custom_command(TARGET 00_smart_app_chat POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/../config
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../config/chat_client_config.xml
        ${CMAKE_CURRENT_BINARY_DIR}/../config/chat_client_config.xml
    COMMENT "Copying 00_smart_app chat client configuration")

if(LC_INSTALL)
    install(TARGETS 00_smart_app_chat
            RUNTIME DESTINATION bin)
endif()
