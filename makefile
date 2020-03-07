clean:
	rm -rf data/raw/*
download:
	python3 src/data/download.py $@
all: download

.PHONY: all clean