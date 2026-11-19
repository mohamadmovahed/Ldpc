add_executable(ldpc_tests
    test_main.cpp
    test_alist.cpp
    test_bch.cpp
    test_decoder.cpp
    test_dvb.cpp
    test_encoder.cpp
    test_sparse_matrix.cpp
)
target_link_libraries(ldpc_tests PRIVATE ldpc::ldpc)
# The DVB test checks the generated tables against their reference statistics.
target_include_directories(ldpc_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME ldpc_tests COMMAND ldpc_tests)
