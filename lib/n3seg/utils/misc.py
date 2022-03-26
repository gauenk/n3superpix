
def optional(pydict,key,default):
    if pydict is None: return default
    elif not(key in pydict): return default
    else: return pydict[key]
