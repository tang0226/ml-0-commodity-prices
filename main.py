import pandas as pd

# ignore pandas performance warnings to avoid flooding the console output
from warnings import simplefilter
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


##################
## APP SETTINGS ##
##################

all_commodities = ['Brent Crude Oil', 'Cocoa', 'Coffee', 'Copper', 'Corn', 'Cotton', 'Crude Oil', 'Feeder Cattle', 'Gold', 'Heating Oil', 'Lean Hogs', 'Live Cattle', 'Lumber', 'Natural Gas', 'Oat', 'Palladium', 'Platinum', 'RBOB Gasoline', 'Silver', 'Soybean Meal', 'Soybean Oil', 'Soybean', 'Sugar', 'Wheat']

# names of datasets to load; must match csv file names
commodities = ['Crude Oil', 'Gold', 'Silver', 'Corn']

# training data settings
max_interpolate = 5
min_training_chunk_size = 365

# feature to predict
Y_commodity = 'Gold'
Y_market_dp = 'Close'
Y_column = f'{Y_commodity} {Y_market_dp}'

# lag feature count
lag_features = 7

# length of walk-forward validation window
chunks_per_training_split = 2

print('App settings:')
print(f' Commodities chosen: {', '.join(commodities)}')
print(f' Feature to predict: {Y_column}')
print(f' Max allowed interpolation window: {max_interpolate} days')
print(f' Min required training chunk size: {min_training_chunk_size} days')
print(f' Lag feature count: {lag_features}')
print(f' Data chunks per training split: {chunks_per_training_split}')

#####################
## DATASET LOADING ##
#####################

print('\nLoading commodity price datasets...')

# Path to csv file folder
folder_path = './Commodity Data'

market_dp_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


comm_cols = [f'{i} {j}' for i in commodities for j in market_dp_names]

# load commodity datasets
dfs = {}
for file_name in commodities:
  # read the csv to a DataFrame and convert dates to datetimes
  comm_df = pd.read_csv(f'{folder_path}/{file_name}.csv')
  comm_df['Date'] = pd.to_datetime(comm_df['Date'])
  dfs[file_name] = comm_df

######################
######################
##  PREPROCESSING   ##
######################
######################

print('\nFinding largest date range present in all datasets...')
# get start and end dates for the largest range included by all commodity datasets
training_start_date = max([dfs[comm]['Date'].min() for comm in commodities])

training_end_date = min([dfs[comm]['Date'].max() for comm in commodities])

# index values for all days within the training range
training_date_indices = pd.date_range(training_start_date, training_end_date, freq='D')
data_points_count = len(training_date_indices)

def get_ymd(ts):
  return ts.strftime('%Y-%m-%d')

print('',
  get_ymd(training_start_date),
  '-',
  get_ymd(training_end_date),
  f'({data_points_count} days)',
)

print('\nClipping and combining commodity price data...')

# main df
df = pd.DataFrame(index=training_date_indices)
for comm in commodities:
  ds_row_count = len(dfs[comm])
  # make Date column the index
  dfs[comm] = dfs[comm].set_index('Date')

  # clip the data range to the common training date range
  dfs[comm] = dfs[comm].loc[
    (dfs[comm].index >= training_start_date) &
    (dfs[comm].index <= training_end_date)
  ]

  record_count = len(dfs[comm])

  # reindex using all target days;
  # missing days will be filled with NaN
  dfs[comm] = dfs[comm].reindex(training_date_indices)

  null_val_ct = dfs[comm].isna().sum().sum()
  null_val_pct = round(null_val_ct / (len(market_dp_names) * data_points_count) * 100, 1)

  print(f' {comm}:')
  print(f'  {null_val_pct}% missing days in dataset')
  print(f'  {record_count} / {ds_row_count} records fall within date range')


  # prefix columns with the commodity name and
  # add the data to the main DataFrame
  dfs[comm] = dfs[comm].add_prefix(comm + ' ')
  df = pd.concat([df, dfs[comm]], axis=1)

# mark fully original data points that will have no interpolated values
df['is_full'] = ~df.isna().any(axis=1)


#######################
### SELECTING DATA  ###
#######################

def get_valid_training_blocks(df, max_interpolate, min_block_size):
  valid_blocks = []
  valid_block_indices = []
  block_start = None
  block_length = 0
  invalid_streak = 0
  last_valid_index = None
  for day_index in list(df.index):
    if not df['is_full'][day_index]:
      invalid_streak += 1
      block_length += 1
      if (invalid_streak >= max_interpolate):
        if block_length >= min_block_size:
          valid_blocks.append(df[block_start : last_valid_index].copy())
        block_length = 0
    else:
      invalid_streak = 0
      block_length += 1
      last_valid_index = day_index
      if block_length == 1:
        block_start = day_index
  if block_length:
    block_length -= invalid_streak
    if block_length >= min_block_size:
      valid_blocks.append(df[block_start : last_valid_index].copy())
  return valid_blocks

# Get training blocks that satisfy the requirements for max interpolation and min split size
print('\nGetting valid training blocks...')
valid_blocks = get_valid_training_blocks(df, max_interpolate, min_training_chunk_size)

######################
### INTERPOLATION  ###
######################
# interpolate  all commodity data and extract the valid blocks
print('\nInterpolating training blocks...')
for b in valid_blocks:
  b.interpolate(method='linear', inplace=True)


# All data columns. will include comm_cols, lag features, and is_full
all_columns = comm_cols + ['is_full']


#####################
### LAG FEATURES  ###
#####################
print(f'\nAdding {lag_features} lag features...')
# lag commodity price points as well as which records are full
for col in comm_cols + ['is_full']:
  for lag in range(1, lag_features + 1):
    lag_col = f'{col} lag_{lag}'
    
    # add columns to the training input list
    all_columns.append(lag_col)

    # Add the lag column to each block
    for b in valid_blocks:
      b[lag_col] = b[col].shift(lag)


# X_columns will be used as input for training and testing
X_columns = all_columns.copy()

# don't train/test on any current values (only lag features)
X_columns.remove('is_full')
for col in market_dp_names:
  for comm in commodities:
    X_columns.remove(f'{comm} {col}')


# remove rows with null lag features
for i in range(len(valid_blocks)):
  valid_blocks[i] = valid_blocks[i].iloc[lag_features:]

block_indices = []
for b in valid_blocks:
  block_indices.append((b.index[0], b.index[-1]))

# combine all blocks into a single Data Frame (will have time gaps, but will be referenced via date indices)
combined_blocks = pd.DataFrame()
for b in valid_blocks:
  combined_blocks = pd.concat([combined_blocks, b])

# Reorder block columns to match all_columns
for i in range(len(valid_blocks)):
  valid_blocks[i] = valid_blocks[i][all_columns]


#######################
### DATA CHUNK CREATION  ###
#######################

# gets index ranges for dividing df into num_chunks equal parts
def get_divided_indices(df, num_chunks): 
    l = len(df)
    chunks_indices = list()
    chunk_size = l // num_chunks
    m = l % num_chunks
    i = 0
    j = 0
    while i < l:
      chunks_indices.append((df.index[i], df.index[i + chunk_size + int(j<m) - 1]))
      if (j < m):
        i += 1
      i += chunk_size
      j += 1
    return chunks_indices


print('\nDividing training blocks...')

# split all valid blocks into smaller chunks with min_training_chunk_size
chunks = []
i = 1
for block in valid_blocks:
  print(f' Block {i} ({len(block)} days, {get_ymd(block.index[0])} - {get_ymd(block.index[-1])}):')
  split = [
    # avoid data leakage by skipping the first `lag_features` rows of each chunk
    (i[0] + pd.Timedelta(lag_features, unit='D'), i[1]) for i in
    get_divided_indices(block, len(block) // min_training_chunk_size)
  ]
  chunks += split
  print(f'  {len(split)} chunks, ~{(split[0][1] - split[0][0]).days} days/chunk')
  i += 1
print(f' {len(chunks)} chunks created')


################################################
### DATA SPLIT CREATION AND STANDARDIZATION  ###
################################################

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
columns_to_standardize = [i for i in all_columns if not i.startswith('is_full')]

# get correct order of columns for the sparse matrix returned by the preprocessor
preprocessed_columns = columns_to_standardize + [i for i in all_columns if i.startswith('is_full')]

standardize_preprocessor = ColumnTransformer(
  transformers=[
    # don't standardize boolean is_full values
    ('scaler', StandardScaler(), columns_to_standardize)
  ],
  remainder='passthrough'
)

print('\nCreating and standardizing training / testing splits...')

splits = []

for i in range(len(chunks) - chunks_per_training_split):
  tr = combined_blocks.loc[chunks[i][0] : chunks[i + chunks_per_training_split - 1][1]]
  # standardize training split
  tr_index = tr.index
  tr = standardize_preprocessor.fit_transform(tr)
  tr = pd.DataFrame(tr, index=tr_index, columns=preprocessed_columns)
  te_chunk = chunks[i + chunks_per_training_split]
  te = combined_blocks.loc[te_chunk[0] : te_chunk[1]]
  # standardize training split
  te_index = te.index
  te = standardize_preprocessor.fit_transform(te)
  te = pd.DataFrame(te, index=te_index, columns=preprocessed_columns)

  splits.append({
    'train': tr,
    'test': te,
  })

print(f' Created {len(splits)} splits, {chunks_per_training_split} chunks/training split')



###########################
### TRAINING / TESTING  ###
###########################
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

regressor = RandomForestRegressor()

print('\nTraining...')

for i in range(len(splits)):
  tr = splits[i]['train']
  te = splits[i]['test']
  
  # when training and testing, select only full rows
  X_train = tr[X_columns + ['is_full']]
  X_train = X_train.loc[X_train['is_full'] == 1][X_columns]

  # get both Y_column and is_full; filter by is_full and select only Y_column
  Y_train = tr[[Y_column, 'is_full']]
  Y_train = Y_train.loc[Y_train['is_full'] == 1][Y_column]
  
  regressor.fit(X_train, Y_train)

  # repeat on test split
  X_test = te[X_columns + ['is_full']]
  X_test = X_test.loc[X_test['is_full'] == 1][X_columns]

  Y_test = te[[Y_column, 'is_full']]
  Y_test = Y_test.loc[Y_test['is_full'] == 1][Y_column]
  
  r2_score = regressor.score(X_test, Y_test)
  mse = mean_squared_error(Y_test, regressor.predict(X_test))

  print(f' Split {i + 1} (train {get_ymd(tr.index[0])} - test {get_ymd(te.index[0])} - {get_ymd(te.index[-1])}): R2-score = {round(r2_score, 4)}, MSE = {round(mse, 4)}')
  
  # plot final split results
  if i == len(splits) - 1:
    plt.title(f'Final Testing Split Prediction (R2-score: {round(r2_score, 4)})')
    plt.xlabel('Date')
    plt.ylabel(f'z-score ({Y_column})')
    plt.plot(te.index, te[Y_column], c='blue', label='actual')
    plt.plot(te.index, regressor.predict(te[X_columns]), c='red', label='predicted')
    plt.legend()
    plt.show()
