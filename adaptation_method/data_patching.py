from data.echosounder_data.preprocessing.resize_and_crop import SplitResizeEchogram
from data.echosounder_data.load_data.get_echograms import get_echogram

import numpy as np

from data.echosounder_data.load_data.get_echograms import get_echograms, count_classes_in_echograms
# You can select a specific year or use 'all' to include multiple years.
years = 2014  
minimum_shape = 224
tuple_frequencies = (18, 38, 70, 120, 200, 333)
echograms = get_echograms(years=years, tuple_frequencies=tuple_frequencies, minimum_shape=minimum_shape) 
print(f"Number of echograms: {len(echograms)}")


