import pandas as pd

print('Using Python to test price-column detection')
print('pandas version:', getattr(pd, '__version__', 'unknown'))

path = 'nyu_prices.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
candidates = [c for c in df.columns if 'discount' in c.lower() and 'charg' in c.lower()]
# replicate header-detection logic from the app
header_row = None
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for i in range(50):
        line = f.readline()
        if not line:
            break
        l = line.lower()
        if 'description' in l and ('discount' in l or 'charge' in l or 'charg' in l):
            header_row = i
            break

if header_row is not None:
    df = pd.read_csv(path, header=header_row)
else:
    df = pd.read_csv(path)
df.columns = df.columns.str.strip()

candidates = [c for c in df.columns if 'discount' in c.lower() and 'charg' in c.lower()]
print('candidates:', candidates)
if candidates:
    price_col = candidates[0]
    print('detected price_col:', price_col)
    print('sample values:')
    print(df[price_col].head(10).to_string(index=False))
else:
    print('no price column found')
