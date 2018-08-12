# TODO: Make better makefile
run:
	g++ -std=c++14 -DXTENSOR_ENABLE_ASSERT=1 -o conv conv.cpp && ./conv
