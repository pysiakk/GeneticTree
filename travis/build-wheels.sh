#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/cp36-cp36m/bin; do
    "${PYBIN}/pip" install -r /io/requirements_dev.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
# for PYBIN in /opt/python/cp36-cp36m/bin/; do
#     "${PYBIN}/pip" install genetic-tree --no-index -f /io/wheelhouse
#     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo) # TODO edit this line
# done
