CFLAGS += -Ofast -DMODEL=_binary_model_bin_start
LDFLAGS += -lm

model.safetensors:
	wget https://huggingface.co/gpt2/resolve/75e09b43581151bd1d9ef6700faa605df408979f/model.safetensors

model.bin: model.safetensors
	tail --bytes=+14292 $< > $@

model.json: model.safetensors
	tail --bytes=+9 $< | head --bytes=14283 > $@

model.o: model.bin
	ld -r -b binary -o $@ $<

prog: main.o model.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f prog *.o model.safetensors
