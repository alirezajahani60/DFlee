
#need to install python3-xlrd
import pandas as pd
import numpy as np

xl = pd.ExcelFile('iraq/Round64_Master_List_IDP_2017-2-2_IOM_DTM.xlsx')
names = ('Place id', 'Governorate', 'District', 'Location name in English',
         'Location name in Arabic', 'Latitude', 'Longitude', 'Families',
         'Individuals', 'Anbar', 'Babylon', 'Baghdad', 'Basrah', 'Dahuk',
         'Diyala', 'Erbil', 'Kerbala', 'Kirkuk', 'Missan', 'Muthanna',
         'Najaf', 'Ninewa', 'Qadissiya', 'Salah al-Din', 'Sulaymaniyah',
         'Thi-Qar', 'Wassit', 'Camp', 'Host families', 'Hotel/Motel',
         'Informal settlements', 'Other shelter type', 'Religious building',
         'Rented houses', 'School building', 'Unfinished/Abandoned building',
         'Unknown shelter type', 'Pre-June14', 'June-July14', 'August14',
         'Post September14', 'Post April15', 'Post March16', 'Post 17 October16',
         'Open Street Map', 'Google Map', 'Bing Map')

governorates = ('Anbar', 'Babylon', 'Baghdad', 'Basrah', 'Dahuk', 'Diyala', 'Erbil',
                'Kerbala', 'Kirkuk', 'Missan', 'Muthanna', 'Najaf', 'Ninewa', 'Qadissiya',
                'Salah al-Din', 'Sulaymaniyah', 'Thi-Qar', 'Wassit')

periods = ('Pre-June14', 'June-July14', 'August14', 'Post September14', 'Post April15',
           'Post March16', 'Post 17 October16')

periods_dic = {'Pre-June14':'31-05-2014', 'June-July14':'31-07-2014', 'August14':'31-08-2014',
               'Post September14':'31-03-2015', 'Post April15':'31-03-2016','Post March16':'30-09-2016',
               'Post 17 October16':'31-01-2016'}

df = xl.parse(xl.sheet_names[0],skiprows=4,header=None, names=names)
#print(df['Location name in Arabic'])

families_tot = np.sum(df['Families'])
people_tot = np.sum(df['Individuals'])
ppl_per_fam = people_tot / families_tot

for g_org in governorates:
    print(g_org)
    # choose only rows with valid entries in the governorate
    df_temp = df[np.isfinite(df[g_org])]
    filename = 'iraq/spawn_data/origin_' + g_org + '.csv'
    with open(filename, "w") as file_out:
        for p in periods:
            num_ppl = np.sum(df_temp[p]) * ppl_per_fam
            file_out.write(periods_dic[p] + ',' + str(int(num_ppl)) + '\n')
            #print(periods_dic[p] + ',' + str(int(num_ppl)))

# loop over governorates of origin
    # select only lines with valid entries
    # integrate lines
    # estimate number of people for each period

# loop over governorates of displacement
    # select only lines with valid entries
    # integrate lines
    # estimate number of people for each period


#TODO
# sum over governorate of origin
# create a gov of origin vs period table to generate spawns
# create a gov of displacement vs period to compare simulation

# 1 csv file per city, period (last date) vs number of people.