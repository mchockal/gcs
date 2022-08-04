import pandas as pd
import matplotlib.pyplot as plt

csv_filepath = "/home/meena270593/dataset/interpolated.csv"

df = pd.read_csv(csv_filepath)

df.hist(column='angle')

plt.show()
