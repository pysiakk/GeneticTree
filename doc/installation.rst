.. _installation:

==================
Installation guide
==================

Requirements
============

Currently package is only available on Linux. If you want to use package on
other operating system you can compile it on your own. More in the section
`Compiling`_.

Before installing GeneticTree you have to install Python >= 3.6. Also you need
some other Python packages:

* numpy>=1.19.2
* scipy>=1.5.2
* aenum>=2.2.4

You can install them by:

.. code-block:: bash

  pip install --upgrade numpy scipy aenum

.. _Installing package:

Installing package
==================

Then you can install package:

.. code-block:: bash

  pip install genetic-tree


.. _Compiling:

Compiling
=========

If your system is not supported or you have any problems with installing
package as in `Installing package`_ section you can compile package from source.

First you have to download code and change directory:

.. code-block:: bash

  git clone https://github.com/pysiakk/GeneticTree.git
  cd GeneticTree # bash
  dir GeneticTree # windows

On Windows if you don't want to use command line you can download zip from
`<https://github.com/pysiakk/GeneticTree>`_ (under Code button click Download
ZIP) and then unpack zip.

Then you have to install necessary requirements:

.. code-block:: bash

  pip install -r requirements.txt

And after all you can compile package:

.. code-block:: bash

  python setup.py build_ext --inplace
