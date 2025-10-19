#!/usr/bin/env python
"""
Diagnostic script to test Theano GPU configuration
Run this before running avocado_train.py to verify GPU setup
"""
import os
import sys

print("=" * 60)
print("Theano GPU Configuration Diagnostic")
print("=" * 60)

# Set environment for GPU
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32,exception_verbosity=high"

print("\n1. Checking Python environment...")
print("   Python version:", sys.version)

print("\n2. Checking Theano installation...")
try:
    import theano
    print("   ✓ Theano version:", theano.__version__)
    print("   ✓ Theano config device:", theano.config.device)
    print("   ✓ Theano floatX:", theano.config.floatX)
except ImportError as e:
    print("   ✗ Error: Theano not installed")
    print("   ", str(e))
    sys.exit(1)

print("\n3. Checking GPU backend (pygpu)...")
try:
    import pygpu
    print("   ✓ pygpu version:", pygpu.__version__)
    from pygpu import gpuarray
    print("   ✓ gpuarray available")
except ImportError as e:
    print("   ✗ Error: pygpu not installed")
    print("   ", str(e))
    print("\n   Install with: conda install -c conda-forge pygpu")
    sys.exit(1)

print("\n4. Testing GPU initialization...")
try:
    import theano.gpuarray
    if theano.gpuarray.pygpu_activated:
        print("   ✓ GPU backend activated")
        ctx = theano.gpuarray.get_context(None)
        print("   ✓ GPU context:", ctx.devname)
    else:
        print("   ✗ GPU backend not activated")
        print("   Check THEANO_FLAGS and CUDA installation")
except Exception as e:
    print("   ✗ Error initializing GPU:")
    print("   ", str(e))
    print("\n   Common fixes:")
    print("   - Update pygpu: conda install -c conda-forge pygpu")
    print("   - Check CUDA version compatibility")
    print("   - Try: THEANO_FLAGS='device=cuda,force_device=True'")

print("\n5. Testing simple GPU computation...")
try:
    import theano.tensor as T
    x = T.vector('x')
    y = x * 2
    f = theano.function([x], y)
    import numpy as np
    result = f(np.array([1, 2, 3], dtype=np.float32))
    print("   ✓ GPU computation test passed")
    print("   Result:", result)
except Exception as e:
    print("   ✗ Error in GPU computation:")
    print("   ", str(e))

print("\n6. Checking cuDNN (if available)...")
try:
    from theano.gpuarray.dnn import dnn_available, dnn_version
    if dnn_available():
        print("   ✓ cuDNN available, version:", dnn_version())
    else:
        print("   ⚠ cuDNN not available (optional, but recommended for better performance)")
except:
    print("   ⚠ cuDNN check failed (optional)")

print("\n7. Checking Avocado...")
try:
    from avocado import Avocado
    print("   ✓ Avocado imported successfully")
except ImportError as e:
    print("   ✗ Error: Avocado not installed")
    print("   ", str(e))

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)
print("\nIf all checks passed, you can run avocado_train.py")
print("If GPU initialization failed, you may need to:")
print("  1. Update pygpu: conda install -c conda-forge pygpu")
print("  2. Check CUDA toolkit version")
print("  3. Use CPU mode as fallback (avocado_train_cpu.py)")
