# coding : utf8

def TianDi():
    a = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    b = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    c = []
    for i in range(60):
        c.append(a[i % len(a)] + b[i % len(b)])
    print(c)
    pass

if __name__ == '__main__':
    TianDi()
    pass


