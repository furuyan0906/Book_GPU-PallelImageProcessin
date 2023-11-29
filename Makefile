
DIR := dummy
ALL_DIRS := $(shell find . -type d | grep -E '*[0-9]{3,}')

.PHONY: dir rm clean

all: dir

dir:
ifneq ($(DIR), dummy)
	@mkdir -p $(DIR)
	@mkdir -p $(DIR)/src
	cp template.mk $(DIR)/Makefile
	cp template.cpp $(DIR)/src/main.cpp
else
	@echo "Usage: make DIR=*"
endif

rm:
ifneq ($(DIR), dummy)
	-@rm -r $(DIR)
else
	@echo "Usage: make rm DIR=*"
endif

clean:
ifneq ($(DIR), dummy)
	make -C $(DIR) clean
else
	@for d in $(ALL_DIRS); do \
		make -C $${d} clean; \
	done
endif

