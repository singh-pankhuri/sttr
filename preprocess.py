import pandas as pd
import numpy as np
import datetime as dt

min_lat = 34.048900
min_long = -120.554200
max_lat = 47.751100
max_long = -105.782100

small = True

def preprocess(dataset):
    if dataset.lower() == 'foursquare':
        df = pd.read_csv('Foursquare/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt', sep='\t', header=None)
        df.columns = ['user','locid','utc','time']
        
        df_poi = pd.read_csv('Foursquare/dataset_WWW2019/raw_POIs.txt', sep='\t', header=None)
        df_poi.columns = ['locid','lat','long','cat','country']
        df_poi = df_poi[['locid','lat','long']]
        df_poi = df_poi[((df_poi.lat >= min_lat) & (df_poi.lat <= max_lat))]
        df_poi = df_poi[((df_poi.long >= min_long) & (df_poi.long <= max_long))]
        reg_list = list(set(df_poi.locid))
        del df_poi

        # Get date in datetime format
        df['utc_clipped'] = df.utc.str.replace(" +0000","", regex = False).str.slice(4) #50s
        df = df[~df.utc_clipped.isin(['1239673244639234'])]
        df['utc_dt'] = pd.to_datetime(df.utc_clipped, format = '%b %d %H:%M:%S %Y') 

        df = df[(df.locid.isin(reg_list))]
        
    elif dataset.lower() == 'gowalla':
        df = pd.read_csv('Gowalla/loc-gowalla_totalCheckins.txt', sep='\t', header=None)
        df.columns = ['user','utc','lat','longi','locid']

        # Get date in datetime format and also get the minimum date
        df['utc_dt'] = pd.to_datetime(df.utc, format = '%Y-%m-%dT%H:%M:%SZ')
        df = df[((df.lat >= min_lat) & (df.lat <= max_lat))]
        df = df[((df.longi >= min_long) & (df.longi <= max_long))]
    
    else:
        raise FileNotFoundError(f"Only Foursquare or Gowalla are accepted inputs")
    
    min_date = df.utc_dt.min() - dt.timedelta(minutes = 1)
    # Filter users with more than 5 check-ins
    users = df.groupby('user',as_index=False).agg({'locid':'count'})
    df_selected = df[df.user.isin(users[users.locid > 5].user)]

    if dataset.lower() == 'foursquare':
        df_selected['key'] = df_selected.locid

    elif dataset.lower() == 'gowalla':
        df_selected['key'] = df_selected.locid.astype(str) + '_' + df_selected.lat.astype(str) + '_' + df_selected.longi.astype(str) 

    # Get locations with at least 10 check-ins
    checkins = df_selected.groupby('key', as_index= False).agg({'user':'count'})
    checkins = checkins[checkins.user > 9]
    checkins = checkins.reset_index(drop = True).drop(columns = ['user']).reset_index()
    checkins.columns = ['locid_num', 'key']
    checkins['locid_num'] = checkins.locid_num + 1

    # Filter for locations with atleast 10 check-ins and add the numeric location ids
    df_selected = df_selected.merge(checkins, on=['key'], how = 'inner')
    users = df_selected.groupby('user',as_index = False).agg({'locid':'count'})
    df_selected = df_selected[df_selected.user.isin(users[users.locid > 2].user)]

    # converting date to time in minutes
    df_selected['date_init'] = pd.to_datetime(min_date)
    df_selected['time_in_min'] = (df_selected.utc_dt - df_selected.date_init).dt.total_seconds().div(60).astype(float)
    df_selected['time_in_min'] = df_selected['time_in_min'].round(0).astype(int)

    # Sort the data before storing
    df_selected = df_selected.sort_values(["user","time_in_min"]).reset_index(drop=True)

    # Reassign sequential user ids for the dataset (algo requirement)
    users = df_selected.groupby('user', as_index = False).agg({"locid_num":"count"}).reset_index()
    users.columns = ['new_userid','user','cnt']
    users['new_userid'] += 1
    users = users[['new_userid','user']]
    df_selected = df_selected.merge(users, on=['user'], how = 'inner')
    df_final = df_selected[['new_userid','locid_num','time_in_min']]

    # Convert to numpy and save as .npy file
    np_final = df_final.to_numpy()
    np.save('./data/'+dataset+'.npy', np_final)

    #POI Generation:
    if dataset.lower() == 'foursquare':
        # Load the data, remove the additional columns
        df_poi = pd.read_csv('Foursquare/dataset_WWW2019/raw_POIs.txt', sep='\t', header=None)
        df_poi.columns = ['locid','lat','long','cat','country']
        df_poi = df_poi[['locid','lat','long']]
        df_poi_selected = df_poi.merge(checkins,  on=['locid'], how = 'inner')

    elif dataset.lower() == 'gowalla':
        df_poi_selected = df_selected[['locid_num','lat','longi']].drop_duplicates()
    
    df_poi_selected = df_poi_selected.sort_values('locid_num').reset_index(drop = True)[['locid_num','lat','long']]
    # Convert to numpy and save as .npy file
    np_poi_final = df_poi_selected.to_numpy()
    np.save('./data/'+dataset+'_small_POI.npy', np_poi_final)

if __name__ == '__main__':
    name = "Foursquare"
    preprocess(name)