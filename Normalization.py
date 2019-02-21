from sklearn import preprocessing
import pandas as pd

data={'price':[492,286,487,519,541,429]}
#Min-Max 标准化
price_frame=pd.DataFrame(data)
min_max_normalizer=preprocessing.MinMaxScaler()
scaled_data=min_max_normalizer.fit_transform(price_frame)

price_frame_normalized=pd.DataFrame(scaled_data)
print(data)
print(price_frame)
print(price_frame_normalized)

#Z-Score标准化
scaled_data2=preprocessing.scale(price_frame)
price_frame_normalized2=pd.DataFrame(scaled_data2)

print(scaled_data2)
print(price_frame_normalized2)
