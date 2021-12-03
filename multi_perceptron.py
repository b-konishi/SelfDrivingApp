import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Multi_perceptron:

    def __init__(self, data, output):
        data = np.array(data)
        output = np.array(output)
        print('data_shape', data.shape)
        print('output_shape', output.shape)


        N = len(data)
        x = data.T
        print(x.shape)
        innum = x.shape[0]
        x = np.append(x, np.ones([1,N]), axis=0)

        yc = output.T
        cnum = yc.shape[0]

        print('N',N)
        print('innum',innum)
        print('cnum',cnum)

        w = np.zeros([cnum,innum+1])
        w_all = np.zeros([cnum,innum+1])
        wt = []
        correct = []
        output = []
        epoch = 10
        for e in range(epoch):
            
            for i in range(N):
                yi = np.zeros(cnum)

                idx = np.ravel(np.where(w.dot(x[:,i]) == np.max(w.dot(x[:,i]))))
                yi[random.choice(idx)] = 1

                
                correct.append(yc[0,i])
                output.append(yi[0])

                if not (yc[:,i] == yi).all():
                    yc_idx = np.ravel(np.where(yc[:,i]==1))[0]
                    w[yc_idx,:] += x[:,i]
                    w[idx,:] -= x[:,i]

                    u = np.zeros([cnum,innum+1])
                    u[yc_idx,:] = x[:,i]
                    u[idx,:] = -x[:,i]

                    w_all += (i+1)*u

                    # print('updated w:', w)
                    wt.append(list(w[0,:]))

        self.w_avg = (w - w_all/(N*epoch))+1e-10
        # print('avg:', self.w_avg)

        mesh_size = 20

        wt = np.array(wt)


    def predict(self, data):
        data.append(1)
        out = self.w_avg.dot(data)
        # print(out)
        print('data_shape', len(data))
        print('out_shape', len(out))
        return out
        
    def get_weight_avg(self):
        return self.w_avg


if __name__ == '__main__':

    m = Multi_perceptron()




    




