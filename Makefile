# TODO: Make better makefile
run:
	g++ -std=c++14 -g -I${CONDA_PREFIX}/include -DXTENSOR_ENABLE_ASSERT=1 -o conv conv.cpp && ./conv

