**********************
Search data structures
**********************

Basic search Data structures
============================

.. automodule:: radbm.search
   :members: 

Binary-based data structures
============================

.. automodule:: radbm.search.binary
   :members:

Superset search data structure 
==============================

.. automodule:: radbm.search.superset
   :members:

Search data structure reduction
===============================
 
 
.. automodule:: radbm.search.reduction
   :members:

Creating custom data structures
===============================

BaseSDS is the base class used to construct new data structures. Only one of batch_insert or insert method needs to be overwritten. Same for batch_search, search and batch_itersearch, itersearch. One can implement get_state and set_state to use the save and load methods.

.. automodule:: radbm.search.base
   :members:

To create custom reduction:

.. automodule:: radbm.search.reduction.base
   :members:
