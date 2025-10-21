# DMX Light setup

## Installing OLA

You're gonna need OLA = Open Lighting Architecture. You can install it with:  `./setup.sh`.

Then if you're magically lucky you can run `uv run python allchantest.py` and see things work.  
It'll blast through all the channels and then back to 0.  If lights change, great.  If not, see Troubleshooting below.

## Defining your lights

You can define your lights in a yaml file.  For example named `constellation.yaml`:

```
lights:
  - name: "light1"
    channel: 1
```

## Troubleshooting / Configuring OLA

This can take a while.  You'll need to edit the files in `/etc/ola/*.conf` and run a bunch of diagnostics
including:

```
ola_dev_info
ola_uni_info
ola_dmxconsole
sudo journalctl -u olad -f
```

Try editing the files in `/etc/ola/*.conf` - try setting `ola-usbserial.conf` or `ola-ftdidmx.conf` to `enabled = true` and point them to the USB device as `/dev/serial/by-id/usb-*` and then `sudo systemctl restart olad`.
Maybe set `enabled = false` for everything else.

and you might have to patch a device to a universe like:

```
ola_patch --device 5 --port 0 --universe 0
```

Also make sure the lights are in DMX mode - especially the first one you're plugging in to.