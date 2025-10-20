# Talk to frank

Several components:

- `lights/` - DMX lights
- `ears/` - Whisper for speech recognition
- `voice/` - Voice generation

## DMX setup


### Installing OLA

You're gonna need OLA = Open Lighting Architecture. You can install it with:

```
sudo apt install ola
sudo systemctl enable --now olad
```

### Configuring OLA

This can take a while.  You'll need to edit the files in `/etc/ola/*.conf` and run a bunch of diagnostics
including:

```
ola_dev_info
```

I ended up setting `enabled = false` for everything except `ola-ftdidmx.conf`, and that was was configured as:
```
enabled = true
frequency = 30
device = /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B0010UH8-if00-port0
```

The script `allchantest.py` can help you figure out if you have anything is listening.  It'll blast 
through all the channels and then back to 0.

```
uv run python allchantest.py
```




## Ears

Currently uses `whisper-cpp`.  
TODO: Replace with faster-whisper.

Building whisper-cpp needs CUDA and a lot of RAM.  (At least 24GB - turn on swap!)

```
cd ears
./setup.sh
```

Test it out:
```
# live mic -> text (PulseAudio/ALSA default device)
./build/bin/whisper-cli -m ./models/ggml-base.en.bin --capture --real-time --print-colors
```