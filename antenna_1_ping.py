import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

# voltage_per_adc_value = 0.122070 * 10**-3
# time_interval = 400 * 10**-6

times = []
voltages = []         #an empty list to store the second column
with open('data_67_002.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    c = 1
    for row in reader:
        if c == 6:
            voltage_per_adc_value = float(row[1][:len(row[1]) - 3]) * 10 **-3
        if c == 7:
            time_interval = float(row[1][:len(row[1]) - 3]) * 10**-6
        if c > 9 :
            times.append(float(row[0]))
            voltages.append(float(row[1]))
        c += 1



clean_times = []
clean_voltages = []
for i in range (len(voltages)):
   new_time = abs(times[i] * time_interval)
   new_voltage = abs(voltages[i] * voltage_per_adc_value)
   #new_voltage = voltages[i]
   clean_times.append(new_time)
   clean_voltages.append(new_voltage)


dt = clean_times[1] - clean_times[0]
n = len(clean_times)

xf = rfftfreq(n, d=dt) # Frequencies associated with each samples
yf = rfft(clean_voltages, n)
plt.plot(xf, abs(yf))
plt.show()

points_per_freq = 100 * 10**6
to_cut_off = int(40)

# xf1 = []
# yf1 = []

# for i in range (len(xf1)):
#    if xf[i] > 50:
#       xf1.append(xf[i])
#       yf1.append(yf[i])

# plt.plot(xf1, yf1)
# plt.show()

# yf1 = yf[:to_cut_off+1]
# xf1 = xf[:to_cut_off+1]

# print(type(xf))

plt.xlim(0.0,50)

plt.plot(xf, abs(yf))
plt.show()