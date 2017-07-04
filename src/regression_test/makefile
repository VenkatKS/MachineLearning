.PHONY : clean

# Library Information
LIB_NAME=ml.so
LIB_TOP=../lib/
LIB_PATH_PARENT=../lib/build/

LIB_FULL_PATH=$(LIB_PATH_PARENT)/$(LIB_NAME)

SOURCES = $(shell echo *.cpp)
HEADERS = $(shell echo *.h)
OBJECTS=$(SOURCES:.cpp=.o)

FIKSENGINE_LIBDIR=build
# Uncomment when adding test header files
# FIKSENGINE_INCDIR=../../../../include

TARGET=$(FIKSENGINE_LIBDIR)/testme.o

all: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -r -f $(FIKSENGINE_LIBDIR)
	(cd $(LIB_TOP); make clean)

$(TARGET) : $(OBJECTS)
	make clean
	(cd $(LIB_TOP); make clean)
	(cd $(LIB_TOP); make all)
	mkdir $(FIKSENGINE_LIBDIR)
	g++ $(OBJECTS) $(LIB_FULL_PATH) -o $(TARGET)
