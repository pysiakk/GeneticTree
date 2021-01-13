.. _api:

===
API
===

Interface of high level classes. The user will use only GeneticTree from
genetic_tree and some enums: Mutation (from mutator), Initialization (from
initializer), Metric (from evaluator) and Selection (from selector). Other
classes are stored as private fields in GeneticTree but for advanced user
it may be sometimes helpful to use them directly.

.. toctree::
    :maxdepth: 1

    genetic_tree
    mutator
    crosser
    initializer
    evaluator
    selector
    stopper
