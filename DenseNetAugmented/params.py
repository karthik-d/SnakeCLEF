import os
from multiprocessing import cpu_count

num_workers = cpu_count()
# num_workers = 1

use_metadata = True

batch_size_cnn = 2
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
directories['attemptpath'] = os.path.join(directories['basepath'], 'DenseNetAugmented')
directories['dynamic'] = os.path.join(directories['attemptpath'], 'dynamic')
directories['output'] = os.path.join(directories['attemptpath'], 'output')
directories['dataset'] = os.path.join(directories['basepath'], "Datasets", "SnakeCLEF-2021")
directories['cnn_checkpoint_weights'] = os.path.join(directories['dynamic'], 'cnn_checkpoint_weights')
directories['cnn_effnet_checkpoint_weights'] = os.path.join(directories['dynamic'], 'cnn_effnet_checkpoint_weights')

"""
directories['input'] = os.path.join('..', 'data', 'input')
directories['output'] = os.path.join('..', 'data', 'output')
directories['working'] = os.path.join('..', 'data', 'working')
directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['cnn_models'] = os.path.join(directories['working'], 'cnn_models')
directories['lstm_models'] = os.path.join(directories['working'], 'lstm_models')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
#directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')
directories['lstm_checkpoint_weights'] = os.path.join(directories['working'], 'lstm_checkpoint_weights')
"""

#directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

files = dict()
files['train_dataparams'] = os.path.join(directories['attemptpath'], "microtrain_metadata.csv")
files['val_dataparams'] = os.path.join(directories['attemptpath'], "microval_metadata.csv")
files['cnn_model'] = os.path.join(directories['output'], 'cnn_model.h5')
files['cnn_effnet_model'] = os.path.join(directories['output'], 'cnn_effnet_model.h5')

"""
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')
"""


"""
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
"""
category_count = 772
category_names = [ x for x in range(category_count) ]

country_count = 187+1   # 187 countries + 1 unknown
country_names = ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'French Guiana', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guam', 'Guatemala', 'Guinea', 'Guinea Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hong Kong S.A.R.', 'Hungary', 'India', 'Indian Ocean Territories', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Libya', 'Lithuania', 'Luxembourg', 'Macau S.A.R', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Northern Cyprus', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo', 'Republic of Serbia', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Samoa', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States of America', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, U.S.', 'West Bank', 'Yemen', 'Zambia', 'Zimbabwe', 'unknown']
country_one_hot = [ [0.0 for x in range(y)]+[1.0]+[0.0 for x in range(country_count-y-1)] for y in range(country_count) ]
map_country_one_hot = dict(zip(country_names, country_one_hot))

continent_count = 7+1   # 1 unknown
continent_names = ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'Oceania', 'South America', 'unknown']
continent_one_hot = [ [0.0 for x in range(y)]+[1.0]+[0.0 for x in range(continent_count-y-1)] for y in range(continent_count) ]
map_continent_one_hot = dict(zip(continent_names, continent_one_hot))

num_labels = category_count

#train_data_size = len(list(csv.reader(open(files['dataparams']))))

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)
