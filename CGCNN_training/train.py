import pandas as pd
df = pd.read_csv('band.csv')
prt = pd.DataFrame()
ind, non = 0, 0
for i in range(len(df)):
    bandtype = str(df.iloc[i]['Band type']).strip()
    if bandtype == 'Indirect':
        ind += 1
        prt = pd.concat([prt, pd.DataFrame([[i, 0]])], ignore_index=True)
    else:
        non += 1
        prt = pd.concat([prt, pd.DataFrame([[i, 1]])], ignore_index=True)
print(f"Indirect: {ind}, Non-indirect: {non}")
prt.to_csv('./cif_dir/id_prop.csv', index=False, header=False)
