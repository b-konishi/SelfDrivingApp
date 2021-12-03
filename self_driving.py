import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import random
import multi_perceptron as mp


class SelfDriving:

    ARROW = {'up':82,'right':83,'left':81,'bottom':84}

    def __init__(self, window_title='Self-Driving', window_size=(1024,1024), step_deg=10, car_color=(150,100,0), car_sensor_num=50, road_color=(204,204,204), road_width=30):
        mode = 'test'

        self.drawing = False # true if mouse is pressed
        self.first_click = False
        self.second_click = False

        self.window_title = window_title
        self.window_size = window_size
        self.step_deg = step_deg
        self.car_sensor_num = car_sensor_num
        self.car_color = car_color
        self.road_color = road_color
        self.road_width = road_width
        self.car_size = (road_width//3*2, road_width)

        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, self.draw_circle)

        self.model = mp.Multi_perceptron()

    def get_model(self):
        return self.model

    def rotate_coord(self, vec, deg):
        rad = 2*np.pi * deg/360
        rotate_mat = np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
        return rotate_mat.dot(vec)

    def put_label(self, img, text, width):
        cv2.putText(img, text, (int(width/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

    def make_image(self):
        self.img = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)


    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.first_click:
                self.start_coord = [x,y]
                self.first_click = False
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.second_click:
                    self.second_coord = [x,y]
                    self.second_click = False
                if self.mode == 'test':
                    cv2.circle(self.img, (x,y), self.road_width, self.road_color, -1)
                elif self.mode == 'prediction':
                    cv2.circle(self.img, (x,y), self.road_width, self.road_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_coord = [x,y]
            if self.mode == 'test':
                self.mode = 'training'
            elif self.mode == 'prediction':
                self.mode = 'run'

            # Draw a point
            # cv2.circle(img,(x,y),road_width,(0,0,255),-1)



    def train(self):

        self.first_click = True
        self.second_click = True

        self.make_image()
        img = self.img

        self.mode = 'test'
        while self.mode == 'test':
            cv2.imshow(self.window_title, img)
            k = cv2.waitKey(1) & 0xFF

            if k != 255:
                print('key: ', k)

            elif k == ord('c'):
                img = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)
            elif k == ord('e'):
                sys.exit(0)

        print('Write training map')
        cv2.imwrite('./training_map2.jpg', img)

        origin_img = np.copy(img)

        car_coord = np.array(self.start_coord).astype(float)
        start_vec = np.array(self.second_coord)-np.array(self.start_coord)
        angle = 360*np.arctan2(start_vec[0], -start_vec[1])/(2*np.pi)
        step_car = self.rotate_coord(np.array([0, -1]), angle)

        print('Start training')
        print(car_coord, self.car_size, angle, self.car_color)
        cv2.ellipse(img, (car_coord, self.car_size, angle), self.car_color, -1)
        sensor_input = []
        desired_operation = []
        while self.mode == 'training':

            cv2.imshow(self.window_title, img)
            k = cv2.waitKey(1) & 0xFF

            vec = self.rotate_coord([0,-self.car_size[1]], angle)
            if all(car_coord+vec+step_car <= self.window_size) and all(car_coord+vec+step_car >= [0,0]):
                car_coord += step_car
            cv_car_coord = np.array(car_coord).astype(float)
            img = np.copy(origin_img)
            cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)


            _front_x = np.linspace(cv_car_coord[0]-self.road_width*2, cv_car_coord[0]+self.road_width*2, num=self.car_sensor_num).astype(int)
            _front_y = int(car_coord[1] - self.car_size[1])
            _front = np.array([[x,_front_y] for x in _front_x])
            d = []
            for coord in _front:
                front = self.rotate_coord(coord-car_coord, angle).astype(float)
                front_coord = cv_car_coord+front

                front_coord[1] = np.where(front_coord[1] >= img.shape[0], img.shape[0]-1, front_coord[1])
                front_coord[0] = np.where(front_coord[0] >= img.shape[1], img.shape[1]-1, front_coord[0])
                front_coord = np.where(front_coord < 0, 0, front_coord)
                
                d.append(1 if list(img[int(front_coord[1]),int(front_coord[0])])==list(self.road_color) else 0)
                cv2.circle(img, (front_coord-2).astype(int), 1, (0,255,255), -1)

            sensor_input.append(np.array(d))

            
            # 道の終点に到達したら、trainingモードを終了し、predictionモードへ移行
            if all(car_coord >= np.array(self.end_coord)-self.road_width) and all(car_coord <= np.array(self.end_coord)+self.road_width):
                self.mode = 'prediction'

            # if k == ARROW['right']:
            if k == ord('d'):
                angle += self.step_deg
                if angle > 180:
                    angle = -180 + (abs(angle)-180)
                step_car = self.rotate_coord(step_car, self.step_deg)

                img = np.copy(origin_img)
                cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)
                print('angle: ', angle)

                desired_operation.append(list(np.eye(3)[2]))

            # elif k == ARROW['left']:
            elif k == ord('a'):
                angle -= self.step_deg
                if angle <= -180:
                    angle = 180 - (abs(angle)-180)
                step_car = self.rotate_coord(step_car, -self.step_deg)

                img = np.copy(origin_img)
                cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)
                print('angle: ', angle)

                desired_operation.append(list(np.eye(3)[0]))

            elif k == ord('e'):
                sys.exit(0)

            else:
                desired_operation.append(list(np.eye(3)[1]))


        self.model.train(sensor_input, desired_operation)



    def predict(self):

        print('Make map')
        # print('change map: Mouse-drag')
        # print('same map: N-key')
        self.first_click = True
        self.second_click = True

        self.make_image()
        img = self.img

        while self.mode == 'prediction':
            cv2.imshow(self.window_title, img)
            k = cv2.waitKey(1) & 0xFF

            if k != 255:
                print('key: ', k)

            elif k == ord('n'):
                break
            elif k == ord('e'):
                sys.exit(0)

        origin_img = np.copy(img)

        print('Write prediction map')
        cv2.imwrite('./prediction_map2.jpg', img)

        car_coord = np.array(self.start_coord).astype(float)
        start_vec = np.array(self.second_coord)-np.array(self.start_coord)
        angle = 360*np.arctan2(start_vec[0], -start_vec[1])/(2*np.pi)
        step_car = self.rotate_coord(np.array([0, -1]), angle)

        print('Start prediction')
        cv2.ellipse(img, (car_coord, self.car_size, angle), self.car_color, -1)
        data = []
        outputs = []
        img = np.copy(origin_img)
        time.sleep(1)
        while self.mode == 'run':
            cv2.imshow(self.window_title, img)
            k = cv2.waitKey(1) & 0xFF

            vec = self.rotate_coord([0,-self.car_size[1]], angle)
            if all(car_coord+vec+step_car <= self.window_size) and all(car_coord+vec+step_car >= [0,0]):
                car_coord += step_car
            cv_car_coord = np.array(car_coord).astype(float)
            img = np.copy(origin_img)
            cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)


            _front_x = np.linspace(cv_car_coord[0]-self.road_width*2, cv_car_coord[0]+self.road_width*2, num=self.car_sensor_num).astype(int)
            _front_y = int(car_coord[1] - self.car_size[1])
            _front = np.array([[x,_front_y] for x in _front_x])
            d = []
            for coord in _front:
                front = self.rotate_coord(coord-car_coord, angle).astype(float)
                front_coord = cv_car_coord+front

                front_coord[1] = np.where(front_coord[1] >= img.shape[0], img.shape[0]-1, front_coord[1])
                front_coord[0] = np.where(front_coord[0] >= img.shape[1], img.shape[1]-1, front_coord[0])
                front_coord = np.where(front_coord < 0, 0, front_coord)

                d.append(1 if list(img[int(front_coord[1]),int(front_coord[0])])==list(self.road_color) else 0)
                cv2.circle(img, (front_coord-2).astype(int), 1, (0,255,255), -1)

            data.append(d)


            out = self.model.predict(d)
            outputs.append(out)
            idx = np.ravel(np.where(out == np.max(out)))
            output = np.zeros(len(out))
            output[random.choice(idx)] = 1

            # 道の終点に到達したら、predictionモードを終了
            if all(car_coord >= np.array(self.end_coord)-self.road_width) and all(car_coord <= np.array(self.end_coord)+self.road_width):
                break

            if all(output == list(np.eye(3)[2])):
                angle += self.step_deg
                if angle > 180:
                    angle = -180 + (abs(angle)-180)
                step_car = self.rotate_coord(step_car, self.step_deg)

                img = np.copy(origin_img)
                cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)


            elif all(output == list(np.eye(3)[0])):
                angle -= self.step_deg
                if angle <= -180:
                    angle = 180 - (abs(angle)-180)
                step_car = self.rotate_coord(step_car, -self.step_deg)

                img = np.copy(origin_img)
                cv2.ellipse(img, (cv_car_coord, self.car_size, angle), self.car_color, -1)
                print('angle: ', angle)


            if k == ord('e'):
                print('break')
                break

        return data, outputs


if __name__ == '__main__':


    driving = SelfDriving(car_sensor_num=10, road_width=30)

    driving.train()
    weight = driving.get_model().get_weight_avg()
    predicted_data, predicted_outputs = driving.predict()

    print('predicted_data: ', np.array(predicted_data).shape)
    cv2.destroyAllWindows()

    plt.figure(figsize=(20,10))
    plt.pcolor(np.flip(weight, axis=0))
    plt.colorbar()
    plt.savefig('weight.png')

    plt.figure(figsize=(20,12))
    plt.rcParams["font.size"] = 18
    plt.subplot(2,1,1)
    plt.imshow(np.array(predicted_data).T)
    plt.title('(a)')
    plt.xlabel('time step')
    plt.ylabel('input units')

    plt.subplot(2,1,2)
    predicted_outputs = np.array(predicted_outputs)
    plt.plot(predicted_outputs[:,0], label='left')
    plt.plot(predicted_outputs[:,1], label='center')
    plt.plot(predicted_outputs[:,2], label='right')
    plt.title('(b)')
    plt.xlabel('time step')
    plt.ylabel('output units')
    plt.legend()

    plt.tight_layout()
    plt.savefig('graph.png')
    plt.show()


