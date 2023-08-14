'''
@Author: haozhic
@Date: 2022-07
@Target: 实现对DEA分析的简单工作


'''

import gurobipy # 这个gurobipy有很多自带的函数，需要阅读说明文档
import pandas as pd
import openpyxl

# 分页显示数据, 设置为 False 不允许分页
pd.set_option('display.expand_frame_repr', False)

# 最多显示的列数, 设置为 None 显示全部列
pd.set_option('display.max_columns', None)

# 最多显示的行数, 设置为 None 显示全部行
pd.set_option('display.max_rows', None)


class DEA(object):
    def __init__(self, DMUs_Name, X, Y, AP=False):
        '''
        :param DMUs_Name:
        :param X:
        :param Y:
        :param AP:
        '''
        self.m1, self.m1_name, self.m2, self.m2_name, self.AP = X.shape[1], X.columns.tolist(), Y.shape[
            1], Y.columns.tolist(), AP
        ''':arg
        m1: 投入项的个数
        m1_name: 投入项的名称
        m2：产出项的个数
        m2_name：产出项名称
        AP：
        '''
        self.DMUs, self.X, self.Y = gurobipy.multidict(
            {DMU: [X.loc[DMU].tolist(), Y.loc[DMU].tolist()] for DMU in DMUs_Name}) #一键多值字典。其实其结构类似于一个复合多类型数据的列表
        '''
        DMUs - [0] : index，决策单元，每一行就是一个决策单元！
        X - [1] : X 投入项的字典
        Y - [2] : Y 产出项的字典
        '''
        print(f'DEA(AP={AP}) MODEL RUNING...')

    def __CCR(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            OE, lambdas, s_negitive, s_positive = MODEL.addVar(), MODEL.addVars(self.DMUs), MODEL.addVars(
                self.m1), MODEL.addVars(self.m2)
            MODEL.update()
            MODEL.setObjectiveN(OE, index=0, priority=1)
            MODEL.setObjectiveN(-(sum(s_negitive) + sum(s_positive)), index=1, priority=0)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) + s_negitive[
                    j] == OE * self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) - s_positive[
                    j] == self.Y[k][j] for j in range(self.m2))
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()
            self.Result.at[k, ('效益分析', '综合技术效益(CCR)')] = MODEL.objVal
            self.Result.at[k, ('规模报酬分析',
                               '有效性')] = '非 DEA 有效' if MODEL.objVal < 1 else 'DEA 弱有效' if s_negitive.sum().getValue() + s_positive.sum().getValue() else 'DEA 强有效'
            self.Result.at[k, ('规模报酬分析',
                               '类型')] = '规模报酬固定' if lambdas.sum().getValue() == 1 else '规模报酬递增' if lambdas.sum().getValue() < 1 else '规模报酬递减'
            for m in range(self.m1):
                self.Result.at[k, ('差额变数分析', f'{self.m1_name[m]}')] = s_negitive[m].X
                self.Result.at[k, ('投入冗余率', f'{self.m1_name[m]}')] = 'N/A' if self.X[k][m] == 0 else s_negitive[m].X / \
                                                                                                     self.X[k][m]
            for m in range(self.m2):
                self.Result.at[k, ('差额变数分析', f'{self.m2_name[m]}')] = s_positive[m].X
                self.Result.at[k, ('产出不足率', f'{self.m2_name[m]}')] = 'N/A' if self.Y[k][m] == 0 else s_positive[m].X / \
                                                                                                     self.Y[k][m]
        return self.Result

    def __BCC(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            TE, lambdas = MODEL.addVar(), MODEL.addVars(self.DMUs)
            MODEL.update()
            MODEL.setObjective(TE, sense=gurobipy.GRB.MINIMIZE)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) <= TE *
                self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) >= self.Y[k][j]
                for j in range(self.m2))
            MODEL.addConstr(gurobipy.quicksum(lambdas[i] for i in self.DMUs if i != k or not self.AP) == 1)
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()

            self.Result.at[
                k, ('效益分析', '技术效益(BCC)')] = MODEL.objVal if MODEL.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
        return self.Result

    def __CRS(self):
        E = []
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            v,u = {},{}
            for i in range(self.m1):
                v[k,i] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name="v_%s%d"%(k,i),lb=0.0001)
            for j in range(self.m2):
                u[k,j] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name="u_%s%d"%(k,j),lb=0.0001)
            MODEL.update()
            MODEL.setObjective(gurobipy.quicksum(u[k,j]*self.Y[k][j] for j in range(self.m2)),gurobipy.GRB.MAXIMIZE)
            MODEL.addConstr(gurobipy.quicksum(v[k,i]*self.X[k][i] for i in range(self.m1))==1)
            for r in self.DMUs:
                MODEL.addConstr(gurobipy.quicksum(u[k,j]*self.Y[r][j] for j in range(self.m2))-gurobipy.quicksum(v[k,i]*self.X[r][i] for i in range(self.m1))<=0)
            MODEL.setParam('OutputFlag',0) # 不让求解过程进行输出！
            MODEL.optimize()

            print(f"The efficiency of DMU {k} is {MODEL.objVal}") # 验证在另一个数据集下面是有好的结果

            # self.Result.at[k,('效益分析','总体效率值(OE)')] = MODEL.objVal
            # E[k] = f"The efficiency of DMU {k} : {MODEL.objVal}"
        # return self.Result

    def __VRS(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model('VRS')
            v,u,u0 = {},{},{}
            for i in range(self.m1):
                v[k,i] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"v_{k}{i}",lb=0.0001)
            for j in range(self.m2):
                u[k,j] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"u_{k}{j}",lb=0.0001)
            u0[k] = MODEL.addVar(vtype=gurobipy.GRB.CONTINUOUS,name=f"u_0{k}",lb=-1000)
            MODEL.update()
            MODEL.setObjective(gurobipy.quicksum(u[k,j]*self.Y[k][j] for j in range(self.m2))-u0[k],gurobipy.GRB.MAXIMIZE)
            MODEL.addConstr(gurobipy.quicksum(v[k,i] * self.X[k][i] for i in range(self.m1))==1)
            for r in self.DMUs:
                MODEL.addConstr(gurobipy.quicksum(u[k,j]*self.Y[r][j] for j in range(self.m2))-gurobipy.quicksum(v[k,i]*self.X[r][i] for i in range(self.m1))-u0[k] <= 0)
            MODEL.setParam('OutputFlag',0)
            MODEL.optimize()

            print(f"The efficiency of DMU {k} is {MODEL.objVal}")
            print(f"The u0{u0[k].varName} : {u0[k].X}")

        return None

    def dea(self):
        # columns_Page = ['效益分析'] * 4 + ['规模报酬分析'] * 2 + ['差额变数分析'] * (self.m1 + self.m2) + ['投入冗余率'] * self.m1 + [
        #     '产出不足率'] * self.m2
        # columns_Group = ['技术效益(BCC)', '规模效益(CCR/BCC)', '综合技术效益(CCR)','总体效率值(OE)', '有效性', '类型'] + (self.m1_name + self.m2_name) * 2
        # '''
        # columns_page : 代表的是第一层的columns
        # columns_group ： 代表的是第二层的columns
        # '''
        #
        # self.Result = pd.DataFrame(index=self.DMUs, columns=[columns_Page, columns_Group])
        # self.__CCR()
        # self.__BCC()
        self.__CRS() # 新增加的模型
        self.__VRS() # 新增加的模型
        # self.Result.loc[:, ('效益分析', '规模效益(CCR/BCC)')] = self.Result.loc[:, ('效益分析', '综合技术效益(CCR)')] / self.Result.loc[:,
        #                                                                                               ('效益分析',
        #                                                                                     '技术效益(BCC)')]
        # return self.Result

    def analysis(self, file_name=None):
        Result = self.dea()
        # file_name = 'DEA 数据包络分析报告.xlsx' if file_name is None else f'\\{file_name}.xlsx'
        # Result.to_excel(file_name, 'DEA 数据包络分析报告')


'''
测试主函数
'''
def main():

    # data = pd.DataFrame({1990: [14.40, 0.65, 31.30, 3621.00, 0.00], 1991: [16.90, 0.72, 32.20, 3943.00, 0.09],
    #                      1992: [15.53, 0.72, 31.87, 4086.67, 0.07], 1993: [15.40, 0.76, 32.23, 4904.67, 0.13],
    #                      1994: [14.17, 0.76, 32.40, 6311.67, 0.37], 1995: [13.33, 0.69, 30.77, 8173.33, 0.59],
    #                      1996: [12.83, 0.61, 29.23, 10236.00, 0.51], 1997: [13.00, 0.63, 28.20, 12094.33, 0.44],
    #                      1998: [13.40, 0.75, 28.80, 13603.33, 0.58], 1999: [14.00, 0.84, 29.10, 14841.00, 1.00]},
    #                     index=['政府财政收入占 GDP 的比例/%', '环保投资占 GDP 的比例/%', '每千人科技人员数/人', '人均 GDP/元', '城市环境质量指数']).T
    #
    # X = data[['政府财政收入占 GDP 的比例/%', '环保投资占 GDP 的比例/%', '每千人科技人员数/人']]  #截取的dataframe
    # Y = data[['人均 GDP/元', '城市环境质量指数']]
    #
    # dea = DEA(DMUs_Name=data.index, X=X, Y=Y)
    # dea.analysis()  # dea 分析并输出表格
    # print(dea.dea())  # dea 分析，不输出结果

    "用于测试CRS模型的数据"
    data = pd.DataFrame({'A':[11,14,2,2,1],'B':[7,7,1,1,1],'C':[11,14,1,1,2],'D':[14,14,2,3,1],'E':[14,15,3,2,3]},
                        index=['x1','x2','y1','y2','y3']).T
    X = data[['x1','x2']]
    Y = data[['y1','y2','y3']]

    dea = DEA(DMUs_Name=data.index,X=X,Y=Y)
    dea.analysis()



if __name__ == '__main__':
    main()
