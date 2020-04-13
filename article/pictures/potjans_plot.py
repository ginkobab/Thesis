import pandas as pd
import matplotlib.pyplot as plt

d = {
'values' : [0.86, 4.45, 7.59, 1.09, 0.3, 1.4, 2.5, 0.5, 0.17, 2.27, 2.46, 0.69],
'authors' :['Potjans and Diesmann (2014)', 'Potjans and Diesmann (2014)', 'Potjans and Diesmann (2014)','Potjans and Diesmann (2014)',  'de Kock and Sakmann (2009)',  'de Kock and Sakmann (2009)',  'de Kock and Sakmann (2009)' , 'de Kock and Sakmann (2009)', 'Optimized Model', 'Optimized Model', 'Optimized Model', 'Optimized Model'],
'layers' : ['L2/3e', 'L4e', 'L5e', 'L6e', 'L2/3e', 'L4e', 'L5e', 'L6e', 'L2/3e', 'L4e', 'L5e', 'L6e']}



df = pd.DataFrame(d)
 
sns.set(font_scale=1.5)
plt.style.use('seaborn-white')
sns.set_palette(sns.light_palette("black"))

fig, ax = plt.subplots(figsize=(16,10))
sns.barplot(x='authors', y='values', hue='layers', data=df, ax=ax)

fig.savefig('potjans_barplot.svg', format='svg')
