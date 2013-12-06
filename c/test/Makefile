# A generic Makefile for building C/C++ projects
# created by - Vishal Patil (vpatil@cs.clemson.edu)
CC=gcc
CPP=g++
AR = ar
GFLAGS =
GET = get
ASFLAGS =
MAS = mas
AS = as
FC = f77
CFLAGS = -c -Wall -g 
INC = -I./headers
LDFLAGS =  
LD = ld
OUTPUT_DIR = ./bin
LIB_DIR = ./lib
LIB = -L$(LIB_DIR) 
LIB_NAME = libcustom 
LFLAGS = -lcurses
LEX = lex
YFLAGS =
YACC = yacc
LOADLIBS =
MAKE = make
MAKEARGS = 'SHELL=/bin/sh'
SHELL = /bin/sh
MAKEFLAGS = b
VPATH=./src:./headers


#Define the source files here
SRC_DIR=./src
SRC = $(wildcard $(SRC_DIR)/*.c $(LIB_DIR)/*.c $(SRC_DIR)/*.cpp)

#Define the object files here
OBJ = $(patsubst %.c,%.o,$(patsubst %.cc,%.o,$(SRC)))	
LIBOBJ = vp_string.o

#Define the output file nae
OUTPUT = main
	
#Create the executable
all : $(LIBOBJ) $(OUTPUT).o
	$(CC) $(OUTPUT).o $(LIBOBJ) $(LIB) $(LFLAGS) -o $(OUTPUT)
	mv $(OUTPUT) $(OUTPUT_DIR)
	mv *.o $(OUTPUT_DIR) 
	echo "Created executable!!"	
	echo $(SRC)	


#Clean the executable and object code
clean:
	-rm ./*.o $(OUTPUT_DIR)/*.o
	-rm $(OUTPUT_DIR)/$(OUTPUT)
	-rm $(LIB_DIR)/*.*
	echo "Finished cleaning"	

#Create lib
lib: $(LIBOBJ)
	$(AR) -r $(LIB_NAME) $(LIBOBJ)
	mv $(LIB_NAME) $(LIB_DIR)
	echo "library created"

#Implicit rule to convert .c files to .o files
%.o : %.c
	$(CC) $(CFLAGS) $(INC) $<
	echo "Compiling C files complete completed"

#Implicit rule to convert .c files to .o files
%.o : %.cpp
	$(CPP) $(CFLAGS) $(INC) $< -o $@
	echo "Compiling C++ completed" 
