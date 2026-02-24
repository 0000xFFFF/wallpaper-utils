GCC = clang++
ARGS = 
DEBUG_ARGS = -D DEBUG -g -fno-omit-frame-pointer
RELEASE_ARGS = -Wall -Wextra -s -march=native
LIBS = `pkg-config --cflags --libs opencv4`

PREFIX = /usr/local
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
INCLUDEDIR = $(PREFIX)/include

PALETTE_FILES = src/palette.cpp
GROUPER_FILES = src/grouper.cpp src/utils.cpp
VALIDATOR_FILES = src/validator.cpp src/utils.cpp
DARKSCORE_FILES = src/darkscore.cpp src/utils.cpp
DARKSCORE-SELECT_FILES = src/darkscore-select.cpp src/utils.cpp

palette: $(PALETTE_FILES)
	$(GCC) $(ARGS) $(RELEASE_ARGS) $(LIBS) $(PALETTE_FILES) -o wpu-palette
	
grouper: $(GROUPER_FILES)
	$(GCC) $(ARGS) $(RELEASE_ARGS) $(LIBS) $(GROUPER_FILES) -o wpu-grouper

validator: $(VALIDATOR_FILES)
	$(GCC) $(ARGS) $(RELEASE_ARGS) $(LIBS) $(VALIDATOR_FILES) -o wpu-validator

darkscore: $(DARKSCORE_FILES)
	$(GCC) $(ARGS) $(RELEASE_ARGS) $(LIBS) $(DARKSCORE_FILES) -o wpu-darkscore

darkscore-select: $(DARKSCORE-SELECT_FILES)
	$(GCC) $(ARGS) $(RELEASE_ARGS) $(LIBS) $(DARKSCORE-SELECT_FILES) -o wpu-darkscore-select



debug-palette: $(PALETTE_FILES)
	$(GCC) $(ARGS) $(DEBUG_ARGS) $(LIBS) $(PALETTE_FILES) -o wpu-palette
	
debug-grouper: $(GROUPER_FILES)
	$(GCC) $(ARGS) $(DEBUG_ARGS) $(LIBS) $(GROUPER_FILES) -o wpu-grouper

debug-validator: $(VALIDATOR_FILES)
	$(GCC) $(ARGS) $(DEBUG_ARGS) $(LIBS) $(VALIDATOR_FILES) -o wpu-validator

debug-darkscore: $(DARKSCORE_FILES)
	$(GCC) $(ARGS) $(DEBUG_ARGS) $(LIBS) $(DARKSCORE_FILES) -o wpu-darkscore

debug-darkscore-select: $(DARKSCORE-SELECT_FILES)
	$(GCC) $(ARGS) $(DEBUG_ARGS) $(LIBS) $(DARKSCORE-SELECT_FILES) -o wpu-darkscore-select



debug: debug-palette debug-grouper debug-validator debug-darkscore debug-darkscore-select


install:
	install -m 755 wpu-palette $(BINDIR)
	install -m 755 wpu-grouper $(BINDIR)
	install -m 755 wpu-validator $(BINDIR)
	install -m 755 wpu-darkscore $(BINDIR)
	install -m 755 wpu-darkscore-select $(BINDIR)


clean:
	rm wpu-palette wpu-grouper wpu-validator wpu-darkscore wpu-darkscore-select

all: palette grouper validator darkscore darkscore-select

release: all
