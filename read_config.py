import configparser

def read_config(path):
    config = configparser.ConfigParser()
    
    # Read config file
    config.read(path)
    
    # Access values
    track_lv = config.get('Tracking','level')
    track_exp = config.get('Tracking','exp')
    track_nxsm = config.get('Tracking','nx_sm')
    track_nxrepeat = config.get('Tracking','nx_repeat')
    track_ntsmooth = config.get('Tracking','nt_smooth')
    track_rmax = config.get('Tracking','r_max')
    ML_losscoeff = config.get('ML','losscoeff')
    ML_LWnumcomps = config.get('ML','LW_numcomps')
    ML_SWnumcomps = config.get('ML','SW_numcomps')

    # Return dictionary
    config_values = {
        'track_lv': track_lv,
        'track_exp':track_exp,
        'track_nxsm':track_nxsm,
        'track_nxrepeat':track_nxrepeat,
        'track_ntsmooth':track_ntsmooth,
        'track_rmax':track_rmax,
        'ML_losscoeff':ML_losscoeff,
        'ML_LWnumcomps':ML_LWnumcomps,
        'ML_SWnumcomps':ML_SWnumcomps,
    }
    return config_values
