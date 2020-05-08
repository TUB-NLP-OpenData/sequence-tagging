from typing import Dict, List, Any


def split_data(split:Dict[str,List[int]], data:List[Any]):
    return {
        split_name: [data[i] for i in indizes]
        for split_name, indizes in split.items()
    }

def split_splits(split:Dict[str,List[int]], data_splits:Dict[str,List[Any]]):
    return {
        split_name: [data_splits[split_name][i] for i in indizes]
        for split_name, indizes in split.items()
    }