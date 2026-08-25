import os
import pickle
import pytest
from configs.biosnap import BIOSNAPConfig

@pytest.fixture
def config():
    return BIOSNAPConfig()

def test_raw_data_files_exist(config):
    smi_path = os.path.join(config.data_dir, "drug_smi_raw.pkl")
    seq_path = os.path.join(config.data_dir, "prot_seq_raw.pkl")
    
    assert os.path.exists(smi_path), f"Missing raw drug file: {smi_path}"
    assert os.path.exists(seq_path), f"Missing raw protein file: {seq_path}"
    
    with open(smi_path, 'rb') as f:
        drugs = pickle.load(f)
    assert isinstance(drugs, dict)
    
    with open(seq_path, 'rb') as f:
        proteins = pickle.load(f)
    assert isinstance(proteins, dict)

def test_cv_splits_exist_and_format(config):
    cv_dir = config.random_cv_dir
    assert os.path.exists(cv_dir), f"Missing CV directory: {cv_dir}"
    
    for fold in range(5):
        split_path = os.path.join(cv_dir, f"data_CV{fold}.pkl")
        assert os.path.exists(split_path), f"Missing split file: {split_path}"
        
        with open(split_path, 'rb') as f:
            data = pickle.load(f)
            
        assert isinstance(data, (list, tuple))
        assert len(data) == 3, f"Split {fold} does not contain [train, valid, test]"
        
        train_data, valid_data, test_data = data
        assert isinstance(train_data, list)
        assert isinstance(valid_data, list)
        assert isinstance(test_data, list)
        
        if len(train_data) > 0:
            sample = train_data[0]
            assert len(sample) == 3, "Sample must be (drug_id, target_id, label)"
            assert isinstance(sample[0], str), "drug_id must be string"
            assert isinstance(sample[1], str), "target_id must be string"
            assert isinstance(sample[2], (int, float, str)), "label format"
