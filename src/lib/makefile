.PHONY : clean

CPPFLAGS= -c -std=c++11

HEADERS = $(shell echo *.hpp)
SOURCES    := $(shell find . -name '*.cpp')
SRCDIRS := $(shell find . -name '*.cpp' -exec dirname {} \; | uniq)
OBJECTS=$(SOURCES:.cpp=.o)

ML_LIBDIR=build
ML_INCDIR=include
ML_TOTALINCDIR=nML

# Individual Parts
ML_LINEARALG=LinearAlgebraLibrary
ML_REGCLASS=RegressionClassificationLibrary

TARGET=$(ML_LIBDIR)/ml.a

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -r -f $(ML_LIBDIR)
	rm -r -f $(ML_LIBDIR)/$(ML_INCDIR)

$(TARGET) : $(OBJECTS)
	mkdir $(ML_LIBDIR)
	# Make Machine Learning Library
	g++ $(CPPFLAGS) $(OBJECTS)
	ar rvs $(TARGET) $(OBJECTS)
	mkdir $(ML_LIBDIR)/$(ML_TOTALINCDIR)
	# copy all headers
	# copy linear algebra headers
	cp -a $(ML_LINEARALG)/$(ML_INCDIR)/. $(ML_LIBDIR)/$(ML_TOTALINCDIR)
	cp -a $(ML_REGCLASS)/$(ML_INCDIR)/. $(ML_LIBDIR)/$(ML_TOTALINCDIR)
	cp -a $(HEADERS) $(ML_LIBDIR)/$(ML_TOTALINCDIR)

	