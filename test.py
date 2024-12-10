##for testing if the pipeline init is working properly!

from src.pipeline import DynamicPipeline

# Define test parameters
params = {
    'run_detrend': True,
    'preproc_line_noise_option': 'None',
    'preproc_line_freqs': [50,160],
    'line_freqs': [60, 120, 180],
    'preproc_lp_freq': 'same',
    'lp_freq': None,
    'preproc_hp_freq': 'same',
    'hp_freq': 0.5,
    'rereference_option': 'ar',
    'missing_chs_option': 'interpolate',
    'drop_extra_chs': True,
    'reorder_chs': True,
}

# Create an instance of DynamicPipeline with test parameters
pipeline = DynamicPipeline(**params)

# Check the values
assert pipeline.run_detrend == True
assert pipeline.preproc_line_noise_option == None
assert pipeline.preproc_line_freqs == [50,160]
assert pipeline.preproc_lp_freq == None
assert pipeline.preproc_hp_freq == 0.5
assert pipeline.rereference_option == 'ar'
assert pipeline.missing_chs_option == 'interpolate'
assert pipeline.drop_extra_chs == True
assert pipeline.reorder_chs == True

print("All tests passed!")


# Define test parameters
params = {
    'run_detrend': True,
    'preproc_line_noise_option': 'default',
    'preproc_line_freqs': 'same',
    'line_freqs': [60, 120, 180],
    'preproc_lp_freq': 'same',
    'lp_freq': 100,
    'preproc_hp_freq': 0.1,
    'hp_freq': 0.5,
    'rereference_option': 'ar',
    'missing_chs_option': 'interpolate',
    'drop_extra_chs': True,
    'reorder_chs': True,
}

# Create an instance of DynamicPipeline with test parameters
pipeline = DynamicPipeline(**params)

# Check the values
assert pipeline.run_detrend == True
assert pipeline.preproc_line_noise_option == 'default'
assert pipeline.preproc_line_freqs == [60, 120, 180]
assert pipeline.preproc_lp_freq == 100
assert pipeline.preproc_hp_freq == 0.1
assert pipeline.rereference_option == 'ar'
assert pipeline.missing_chs_option == 'interpolate'
assert pipeline.drop_extra_chs == True
assert pipeline.reorder_chs == True

print("All basic tests passed!")



# Define test parameters
params = {
    'run_detrend': True,
    'preproc_line_noise_option': False,
    'preproc_line_freqs': 'None',
    'line_freqs': [60, 120, 180],
    'preproc_lp_freq': 'None',
    'lp_freq': 100,
    'preproc_hp_freq': 0.1,
    'hp_freq': 0.5,
    'rereference_option': 'None',
    'missing_chs_option': 'interpolate',
    'drop_extra_chs': True,
    'reorder_chs': True,
}

# Create an instance of DynamicPipeline with test parameters
pipeline = DynamicPipeline(**params)

# Check the values
assert pipeline.run_detrend == True
assert pipeline.preproc_line_noise_option == None
assert pipeline.preproc_line_freqs == []
assert pipeline.preproc_lp_freq == None
assert pipeline.preproc_hp_freq == 0.1
assert pipeline.rereference_option == None
assert pipeline.missing_chs_option == 'interpolate'
assert pipeline.drop_extra_chs == True
assert pipeline.reorder_chs == True

print("All basic tests passed!")


