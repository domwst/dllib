CC=clang++
PROD_FLAGS=-Wall -Wextra -O2 -std=c++2a
TEST_FLAGS=-g -DLOCAL -Wall -Wextra -pedantic -std=c++2a -O0 -Wformat=2 -Wfloat-equal -Wconversion -Wcast-qual -Wcast-align -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fsanitize=address -fsanitize=undefined -fstack-protector -lmcheck -Wno-unused-parameter -Wno-unused-result -Wno-shadow

tensor-tests:
	$(CC) tests/TensorTests.cpp -o tests/TensorTests $(TEST_FLAGS)

run-tensor-tests: tensor-tests
	tests/TensorTests

tests: tensor-tests

run-tests: run-tensor-tests

