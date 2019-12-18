import pandas as pd

df = pd.read_excel(r'pgrfilestore.xlsx')
prefixes = list(df['Ulb Code'])
suffixes = list(df['File Storeid'])
for i in range(0, len(suffixes)):
    suffixes[i] = suffixes[i].strip()
complaint = list(df['Complaint Type'])
for i in range(0, len(complaint)):
    complaint[i] = complaint[i].strip()

# print(df.groupby('Complaint Type')['File Storeid'].nunique())
print(df['Complaint Type'].value_counts())

pj_pgr_types = [
    'Street Light Not Working',
    'Overflowing Or Blocked Drain',
    'Block Or Overflowing Sewage',
    'Garbage Needs Tobe Cleared',
    'Others',
    'Damaged Road',
    'Request Spraying Or Fogging Operation',
    'illegal Discharge Of Sewage',
    '',
    '',
    '',
    ''
]

# for i in set(complaint):
#     print(i)
