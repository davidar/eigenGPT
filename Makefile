CXXFLAGS += -Ofast

model.safetensors:
	wget https://huggingface.co/gpt2/resolve/75e09b43581151bd1d9ef6700faa605df408979f/model.safetensors

model.o: model.safetensors
	ld -r -b binary -o $@ $<

prog: main.o model.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f prog *.o model.safetensors
