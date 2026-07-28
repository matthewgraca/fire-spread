import s3fs
import traceback
import pandas as pd

fs = s3fs.S3FileSystem(anon=True)

## noaa-goes18 succeeds, WEST fails
#path = 'WEST/ABI-L2-FDCC/2026/208/16'
path = 'noaa-goes18/ABI-L2-FDCC/2026/208/16'
##

# okay, WEST and EAST are just aliases. brian remaps them.
# doesn't his code always remap to the newest satellite? then why does the new sats not work with these aliases, and why do old ones work..?

## yet WEST succeeds here. my guess is WEST maps to noaa-goes17
#path = 'WEST/ABI-L2-FDCC/2020/245/06'
path = 'noaa-goes18/ABI-L2-FDCC/2020/245/06' # expect missing (doesn't for some reason, even though the folder dne?) if this doesn' get pruned, does the df[sat.. etc] freak out?
# okay so fs.ls explodes when it sees a file path that dne; that's why it skips FNF, it's just exploding and getting caught as a generic error upstream
#Data for 2020-09-01 08:00:00 missing/corrupted for GOES-WEST.

try:
    print('before ls')
    files = fs.ls(path, refresh=True)
    #files = fs.glob(path + '/*.nc', refresh=True)
    print(files)
    print('after ls')
    print('hi') # why does this silently fail with path 2?
    df = pd.DataFrame(files, columns=["file"])
    df.drop(index=df.index[~df["file"].str.contains(".nc")],inplace=True)
    df[["product_mode", "satellite", "start", "end", "creation"]] = (
        df["file"].str.rsplit("_", expand=True, n=5).loc[:, 1:]
    )
    print(df)
except FileNotFoundError as e:
    print(e)
except SystemExit as e:
    print('Caught sys exit')
    print(e)
except Exception as e:
    print('Something else:')
    print(traceback.format_exc())
