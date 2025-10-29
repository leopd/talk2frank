# Eyes

```
uv run framegrab autodiscover
```

If that doesn't work, try:

```
sudo usermod -aG video $USER   # then log out/in
```

Pick your camera config, and put it into `camera.yaml`

## Insultcam

For a 1-off test, just run:

```
uv run python insultcam.py
```

