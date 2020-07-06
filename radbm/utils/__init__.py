def unique_list(it):
    """
    Create a list from an iterable with only unique element and where
    the order is preserved.
    
    Parameters
    ----------
    it : iterable
        Items should be hashable and comparable
        
    Returns : list
        All items in the list is unique and the order of the iterable
        is preserved.
    """
    unique = set()
    return [i for i in it if i not in unique and unique.add(i) is None]