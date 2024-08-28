import configparser
#---------------------------------------------------------------------------------------------------------
# Notes:
# (1) Remove "VMAX" from "SHIPSop_varname" if you would like to set "VMAX" as a target
#---------------------------------------------------------------------------------------------------------
def create_config():
    config = configparser.ConfigParser()

    # Add sections and value pairs
    config['Tracking'] = {'level': 600,
                          'exp':'CTRL',
                          'nx_sm':9, # 2D filter size
                          'nx_repeat':15, # Smooth things spatially for nx_repeat times
                          'nt_smooth':3, # Time-series smoothing
                          'r_max':6, # Mask out grid points beyond [degree] from TC center
                         }
    config['ML'] = {'losscoeff':1,
                   'LW_numcomps':0.5,
                   'SW_numcomps':0.8,
                   }

    # Write the config to a file
    with open('config.ini','w') as configfile:
        config.write(configfile)

if __name__=="__main__":
    create_config()
