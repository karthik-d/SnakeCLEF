import os
from multiprocessing import cpu_count

num_workers = cpu_count()

use_metadata = True

batch_size_cnn = 128
batch_size_lstm = 512
batch_size_eval = 128
metadata_length = 2  # No of features
num_channels = 3
cnn_lstm_layer_length = 2208

target_img_size = (224,224)

image_format = 'jpg'

train_cnn = False
generate_cnn_codes = False
train_lstm = False
test_cnn = False
test_lstm = False

#LEARNING PARAMS
cnn_adam_learning_rate = 1e-4
cnn_adam_loss = 'categorical_crossentropy'
cnn_epochs = 50

lstm_adam_learning_rate = 1e-4
lstm_epochs = 100
lstm_loss = 'categorical_crossentropy'

#DIRECTORIES AND FILES
directories = dict()
directories['basepath'] = os.path.abspath(os.pardir)
directories['dataset'] = os.path.join(directories['basepath'], "Datasets", "SnakeCLEF-2021")
directories['input'] = os.path.join('..', 'data', 'input')
directories['output'] = os.path.join('..', 'data', 'output')
directories['working'] = os.path.join('..', 'data', 'working')
directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['cnn_models'] = os.path.join(directories['working'], 'cnn_models')
directories['lstm_models'] = os.path.join(directories['working'], 'lstm_models')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')
directories['lstm_checkpoint_weights'] = os.path.join(directories['working'], 'lstm_checkpoint_weights')

directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

files = {}
files['dataparams'] = os.path.join(directories['basepath'], "DenseNetAugmented", "microtrain_metadata.csv")
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')


"""
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
"""
category_count = 772
category_names = [ x for x in range(category_count) ]

country_count = 99+1   # 99 countries + 1 unknown
country_names = ['Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Bahamas', 'Belarus', 'Belgium', 'Belize', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Burma', 'Cambodia', 'Canada', 'Cayman Islands', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Czech Republic', 'Denmark', 'East Timor', 'Ecuador', 'El Salvador', 'Ethiopia', 'Finland', 'France', 'French Guiana', 'Georgia', 'Germany', 'Greece', 'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hong Kong S.A.R.', 'India', 'Indonesia', 'Israel', 'Italy', 'Japan', 'Kenya', 'Kosovo', 'Laos', 'Latvia', 'Lebanon', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mexico', 'Mongolia', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'Nicaragua', 'Nigeria', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Republic of Serbia', 'Romania', 'Russia', 'Rwanda', 'Seychelles', 'Singapore', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden', 'Switzerland', 'Taiwan', 'Tanzania', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Ukraine', 'United Kingdom', 'United States of America', 'Uruguay', 'Venezuela', 'Vietnam', 'Zambia', 'Zimbabwe', 'unknown']
country_one_hot = [ [0.0 for x in range(y)]+[1.0]+[0.0 for x in range(country_count-y-1)] for y in range(country_count) ]
map_country_one_hot = dict(zip(country_names, country_one_hot))

continent_count = 7+1   # 1 unknown
continent_names = ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'Oceania', 'South America', 'unknown']
continent_one_hot = [ [0.0 for x in range(y)]+[1.0]+[0.0 for x in range(continent_count-y-1)] for y in range(continent_count) ]
map_continent_one_hot = dict(zip(continent_names, continent_one_hot))

num_labels = category_count

dataset_size = len(list(csv.reader(open(files['dataparams']))))

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)
