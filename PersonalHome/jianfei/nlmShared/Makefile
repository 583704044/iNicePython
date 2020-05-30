all:

clean:
	rm -f *.o *.so *.html

nlmsimple.so: nlmsimple.o
  gcc -shared -o libnlmsimple.so nlmsimple.o

nlmsimple.o: nlm_toy.cpp
  gcc -c -Wall -Werror -fpic nlm_toy.cpp