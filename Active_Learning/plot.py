import matplotlib.pyplot as plt

def plot(index, data, figure_name):

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(index, data)

    # 找出最小值和最后的值
    min_value = min(data)
    last_value = data[-1]
    min_index = data.index(min_value)
    last_index = len(data) - 1

    # 标注最小值（左上角）
    plt.annotate(f'Min: {min_value}', 
                 xy=(min_index, min_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    # 标注最后的值（右上角）
    plt.annotate(f'Last: {last_value}', 
                 xy=(last_index, last_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right',
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    # 设置标题和轴标签
    plt.title(figure_name)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 保存图表
    plt.savefig("./figures/"+figure_name+".png")
    plt.close()


Budget = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]

# data = [0.7755, 0.7722, 0.8058, 0.8226, 0.8276, 0.8337, 0.8384, 0.8355, 0.8365, 0.8463, 0.8443, 0.8489, 0.8485, 0.8482, 0.8466, 0.8494, 0.8492, 0.8462, 0.8523, 0.8471, 0.8531, 0.8478, 0.8475, 0.8479, 0.8411, 0.8362, 0.8352, 0.8326, 0.8347, 0.8329, 0.8322, 0.8288, 0.8287, 0.8234, 0.8060, 0.8210, 0.8233, 0.8123, 0.8149, 0.8045]

# plot(Budget[0:20], data[0:20], "amazon_ori")

data = [0.8244, 0.8297, 0.8458, 0.8465, 0.8455, 0.8455, 0.8477, 0.8452, 0.8497, 0.8471, 0.8499, 0.8530, 0.8493, 0.8561, 0.8440, 0.8459, 0.8434, 0.8502, 0.8457, 0.8479, 0.8479, 0.8540, 0.8512, 0.8461, 0.8460, 0.8495, 0.8415, 0.8443, 0.8404, 0.8458, 0.8476, 0.8441, 0.8476, 0.8411, 0.8453, 0.8489, 0.8458, 0.8507, 0.8463, 0.8422]

plot(Budget[0:20], data[0:20], "amazon_feat")