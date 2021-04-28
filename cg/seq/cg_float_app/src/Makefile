SRCS = main.cpp

solver: $(SRCS)
	c++ -O3 -o $@ $(SRCS) -lm

debug_solver: $(SRCS)
	c++ -g -o $@ $(SRCS) -lm

clean:
	rm -f solver debug_solver
