import matplotlib.pyplot as plt

versions = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12']
miou = [0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83]

plt.figure(figsize=(12,6))
plt.plot(versions, miou, marker='o', linestyle='-', color='blue', linewidth=2)
plt.xlabel('Model Version')
plt.ylabel('mIoU')
plt.title('mIoU Trend Across Versions')
plt.ylim(0, 1)  # mIoU 范围
plt.grid(True)
plt.show()