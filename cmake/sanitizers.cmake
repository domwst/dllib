if (DLLIB_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()
