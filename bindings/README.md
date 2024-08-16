# Python bindings for GATO

```
cd python/gato
```

To build python bindings, run "bindings.sh" in tools/:
```
#from bindings/python/gato
../../../tools/bindings.sh
```

Then source the "setup_gato_env.sh" script to include the bindings in your environment:
```
source <../../../tools/setup_gato_env.sh>
```

You can now use the bindings in Python!
```
python3 sqp_pcg_n.py
```
