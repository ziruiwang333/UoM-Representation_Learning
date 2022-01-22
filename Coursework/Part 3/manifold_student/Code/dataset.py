import numpy as np
import math

def ten_city():
    city_name = ['Atlanta', 'Chicago', 'Denver', 'Houston', 'Los Angeles', 'Miami', 'New York', 'San Francisco', 'Seattle', 'Washington D.C.']
    num_city = len(city_name)
    data = np.zeros([num_city, num_city])
    data[1,0] = 587
    data[2,0] = 1212; data[2,1] = 920
    data[3,0] = 701;  data[3,1] = 940;  data[3,2] = 879
    data[4,0] = 1936; data[4,1] = 1745; data[4,2] = 831;  data[4,3] = 1374
    data[5,0] = 604;  data[5,1] = 1188; data[5,2] = 1726; data[5,3] = 968;  data[5,4] = 2339
    data[6,0] = 748;  data[6,1] = 713;  data[6,2] = 1631; data[6,3] = 1420; data[6,4] = 2451; data[6,5] = 1092
    data[7,0] = 2139; data[7,1] = 1858; data[7,2] = 949;  data[7,3] = 1645; data[7,4] = 347;  data[7,5] = 2594; data[7,6] = 2571
    data[8,0] = 2182; data[8,1] = 1737; data[8,2] = 1021; data[8,3] = 1891; data[8,4] = 959;  data[8,5] = 2734; data[8,6] = 2408; data[8,7] = 678
    data[9,0] = 543;  data[9,1] = 597;  data[9,2] = 1494; data[9,3] = 1220; data[9,4] = 2300; data[9,5] = 923;  data[9,6] = 205;  data[9,7] = 2442; data[9,8] = 2329
    return data, city_name

def synthetic_spiral():
    """
    Spiral data
    """
    sqrt_two = math.sqrt(2)
    data = [ [math.cos(k/sqrt_two), math.sin(k/sqrt_two), k/sqrt_two] for k in range(30)]
    data = np.vstack(data)
    return data.T

def bars():
    file_name = './Data/bars.npz'
    data = np.load(file_name)
    v_bars = data['v_bars'].astype(np.float)
    v_centers = data['v_centers'].astype(np.float)
    h_bars = data['h_bars'].astype(np.float)
    h_centers = data['h_centers'].astype(np.float)
    return np.vstack((v_bars, h_bars)), np.vstack((v_centers, h_centers))

def face_tenenbaum():
    data = np.load('./Data/face_tenenbaum.npz')
    data = data['face']
    return data.T
