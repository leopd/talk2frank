# Talk to frank

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

The script `allchantest.py` can help you figure out if you have anything working.  It'll blast 
through all the channels and then back to 0.

```
uv run python allchantest.py
```



