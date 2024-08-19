DATA_FILES=$(addprefix data/, train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte)

data/%.gz:
	mkdir -p data
	wget -P data https://raw.githubusercontent.com/fgnt/mnist/master/$*.gz

data/%: data/%.gz
	gunzip -c $< > $@

.PHONY: all
all: $(DATA_FILES)

.PHONY: all
run: $(DATA_FILES)
	cargo run --release