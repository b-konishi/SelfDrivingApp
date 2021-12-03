import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultiPerceptron:

    def __init__(self):
        pass


    def train(self, inputs, desired_outputs, epoch=100):

        inputs = np.array(inputs)
        desired_outputs = np.array(desired_outputs)

        print('inputs_shape', inputs.shape)
        print('desired_outputs_shape', desired_outputs.shape)

        N = len(inputs)
        x = inputs.T
        print(x.shape)
        N_in = x.shape[0]
        x = np.append(x, np.ones([1,N]), axis=0)

        yd = desired_outputs.T
        N_out = yd.shape[0]

        print('N', N)
        print('N_in', N_in)
        print('N_out', N_out)

        w = np.zeros([N_out,N_in+1])
        w_all = np.zeros([N_out,N_in+1])
        wt = []
        correct = []
        output = []
        for e in range(epoch):

            print('\repoch: {}/{}'.format(e, epoch), end='')
            
            for i in range(N):
                yi = np.zeros(N_out)

                idx = np.ravel(np.where(w.dot(x[:,i]) == np.max(w.dot(x[:,i]))))
                yi[random.choice(idx)] = 1

                
                correct.append(yd[0,i])
                output.append(yi[0])

                if not (yd[:,i] == yi).all():
                    yd_idx = np.ravel(np.where(yd[:,i]==1))[0]
                    w[yd_idx,:] += x[:,i]
                    w[idx,:] -= x[:,i]

                    u = np.zeros([N_out,N_in+1])
                    u[yd_idx,:] = x[:,i]
                    u[idx,:] = -x[:,i]

                    w_all += (i+1)*u

                    wt.append(list(w[0,:]))

        self.w_avg = (w - w_all/(N*epoch)) + 1e-10
        mesh_size = 20
        wt = np.array(wt)

        print('')


    def predict(self, data):
        data.append(1)
        out = self.w_avg.dot(data)
        return out
        
    def get_weight_avg(self):
        return self.w_avg


if __name__ == '__main__':

    m = MultiPerceptron()




    




