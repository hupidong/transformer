from src.transformer import PositionalEncoding
import matplotlib.pyplot as plt

pe=PositionalEncoding(d_model=512,dropout=0.0,max_len=1024)

# 某个位置的编码向量
y=pe.pe[0,10,:]
plt.plot(y)
plt.show()

plt.imshow(pe.pe.reshape(pe.pe.size(1), pe.pe.size(2)))
plt.show()

pass

