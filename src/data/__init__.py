from .dataset import DTIDataset, collate_fn

# Preprocessing imports are lazy to avoid requiring transformers/esm at import time.
# Import directly from src.data.preprocessing when needed:
#   from src.data.preprocessing import get_drug_graph, get_protbert_embedding, ...
def __getattr__(name):
    _preprocessing_names = {
        'get_drug_graph', 'get_protbert_embedding',
        'target_graph_construct', 'get_target_graph',
        'smi_2_graph', 'one_hot', 'get_uv', 'padding',
    }
    if name in _preprocessing_names:
        from . import preprocessing
        return getattr(preprocessing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
