# CPSC532M_Project

## Developing

`docker compose run shell`

VS Code Extension: Remote Development
`Dev Containers: Attach to running container`


## Workaround I needed

Error: `ImportError: cannot import name 'PILLOW_VERSION' from 'PIL' (/usr/local/lib/python3.11/site-packages/PIL/__init__.py)`

Solution:
This might be late, but for fellow travelers led to this by Google, a simpler workaround for this, instead of downgrading can be done by patching the torchvision/transforms/functional.py file. If you remove the PILLOW_VERSION import from the import and then replace PILLOW_VERSION with PIL.version in line 727, then it works without a downgrade. Official PIL docs recommends using version instead.

https://github.com/pytorch/vision/issues/1712
