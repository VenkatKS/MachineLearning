.PHONY : clean

CPPFLAGS= -c -std=c++11

SOURCES = $(shell echo *.cpp)
HEADERS = $(shell echo *.hpp)
OBJECTS=$(SOURCES:.cpp=.o)

ML_LIBDIR=build
ML_INCDIR=include

INTERIM_TARGET=$(ML_LIBDIR)/ml.o
TARGET=$(ML_LIBDIR)/ml.so

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -r -f $(ML_LIBDIR)

$(TARGET) : $(OBJECTS)
	mkdir $(ML_LIBDIR)
	# Make Machine Learning Library
	g++ $(CPPFLAGS) $(OBJECTS)
	ar rvs $(TARGET) *.o